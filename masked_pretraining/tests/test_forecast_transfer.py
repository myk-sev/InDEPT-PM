import csv
from dataclasses import asdict
import tempfile
import unittest
from pathlib import Path

import torch

from evaluate_bridge_forecast import fit_linear_baseline, linear_prediction, metric_report
from inference.artifacts import artifact_paths, dataset_model_stem
from masked_pretraining.masking import (
    STAGES,
    TEMPO_BRIDGE_MISSING_FRACTIONS,
    TEMPO_BRIDGE_STAGES,
)
from masked_pretraining.models import (
    ModelConfig as HistoryConfig,
    build_model as build_history_model,
    model_names as history_model_names,
)
from masked_pretraining.train import write_masked_final_report
from pm25_models import (
    BridgeForecastConfig,
    bridge_forecast_name,
    build_model,
    validate_bridge_checkpoint,
)
from pm25_transformer import (
    ZScores,
    bridge_history_zscores,
    build_parser,
    load_checkpoint,
    resolve_horizon_schedule,
    save_checkpoint,
    train,
)


def _config() -> BridgeForecastConfig:
    return BridgeForecastConfig(
        history_hours=12,
        prediction_hours=6,
        model_dim=8,
        layers=1,
        heads=2,
        dropout=0,
        forecast_embedding_dim=8,
        forecast_heads=2,
        forecast_head_dim=4,
        forecast_layers=1,
        forecast_feedforward_dim=16,
        forecast_dropout=0,
        decoder_embedding_dim=8,
        decoder_heads=2,
        decoder_head_dim=4,
        decoder_layers=1,
        decoder_feedforward_dim=16,
        decoder_dropout=0,
    )


def _checkpoint(model_name: str, config: BridgeForecastConfig) -> dict:
    history = build_history_model(model_name, config.history_config)
    return {
        "model_state": history.state_dict(),
        "optimizer_state": {},
        "metadata": {
            "model_name": model_name,
            "model_config": asdict(config.history_config),
            "stage": TEMPO_BRIDGE_STAGES[-1],
            "completed_stages": list(STAGES + TEMPO_BRIDGE_STAGES),
            "training_data_sha256": "abc123",
            "normalizer": {
                "mean": [10.0, 4.0],
                "standard_deviation": [5.0, 2.0],
            },
            "masking": {
                "sentinel": -9.0,
                "tempo_missingness_bridge": {
                    "enabled": True,
                    "synthetic_only": True,
                    "tempo_data_used": False,
                    "outdoor_artificial_missing_fractions": dict(
                        TEMPO_BRIDGE_MISSING_FRACTIONS
                    ),
                }
            },
            "transfer": {
                "input_feature_order": [
                    "outdoor_value",
                    "indoor_value",
                    "daily_sin",
                    "daily_cos",
                    "weekly_sin",
                    "weekly_cos",
                    "annual_sin",
                    "annual_cos",
                ]
            },
        },
    }


class ForecastTransferTests(unittest.TestCase):
    def test_artifact_contract_uses_dedicated_inference_folders(self):
        stem = dataset_model_stem(
            Path("k12_excl_fine_t_masked_training_data.csv"),
            "dual-encoder-cross-fusion",
        )
        paths = artifact_paths(stem, Path("inference"))

        self.assertEqual(stem, "k12_excl_fine_t_dual-encoder-cross-fusion")
        self.assertEqual(paths.checkpoint, Path("inference/checkpoints") / f"{stem}.pt")
        self.assertEqual(paths.cache, Path("inference/caches") / f"{stem}.pt")
        self.assertEqual(paths.metrics, Path("inference/metrics") / f"{stem}.csv")
        self.assertEqual(paths.graph, Path("inference/graphs") / f"{stem}.png")
        self.assertEqual(paths.report, Path("inference/reports") / f"{stem}.csv")
        self.assertEqual(
            paths.reconstructions, Path("inference/reconstructions") / stem
        )

    def test_every_history_model_has_a_forecast_counterpart(self):
        config = _config()
        history = torch.randn(2, config.history_hours, 8)
        history[:, 3, 0] = torch.nan
        forecast = torch.randn(2, config.prediction_hours, 7)

        for source_name in history_model_names():
            model = build_model(bridge_forecast_name(source_name), config)
            self.assertEqual(
                tuple(model(history, forecast).shape),
                (2, config.prediction_hours),
            )

    def test_strict_transfer_discards_only_reconstruction_head(self):
        config = _config()
        checkpoint = _checkpoint("dual-encoder-cross-fusion", config)
        model = build_model(
            bridge_forecast_name("dual-encoder-cross-fusion"), config
        )

        transferred = model.load_pretrained_history(checkpoint)

        self.assertTrue(transferred)
        self.assertFalse(any(name.startswith("reconstruction_head.") for name in transferred))
        self.assertEqual(set(transferred), set(model.history.state_dict()))

    def test_transfer_rejects_incomplete_bridge(self):
        config = _config()
        checkpoint = _checkpoint("single-self-attention-encoder", config)
        checkpoint["metadata"]["completed_stages"] = list(STAGES)

        with self.assertRaisesRegex(ValueError, "full curriculum"):
            validate_bridge_checkpoint(checkpoint)

    def test_bridge_normalizer_is_used_only_for_history_encoder(self):
        config = _config()
        checkpoint = _checkpoint("single-self-attention-encoder", config)
        zscores = ZScores(1, 2, 3, 4, 5, 6)

        adjusted = bridge_history_zscores(zscores, checkpoint)

        self.assertEqual((adjusted.indoor_mean, adjusted.forecast_mean), (1, 5))
        self.assertEqual(
            (
                adjusted.encoder_outdoor_mean,
                adjusted.encoder_outdoor_std,
                adjusted.encoder_indoor_mean,
                adjusted.encoder_indoor_std,
            ),
            (10, 5, 4, 2),
        )

    def test_horizon_schedule_is_explicit_and_complete(self):
        self.assertEqual(
            resolve_horizon_schedule(6, 5, [2, 4, 6], [1, 2, 2]),
            [2, 4, 4, 6, 6],
        )
        with self.assertRaisesRegex(ValueError, "sum"):
            resolve_horizon_schedule(6, 4, [2, 6], [1, 2])

    def test_forecast_checkpoint_is_self_contained_after_transfer(self):
        config = _config()
        source = _checkpoint("gru", config)
        model = build_model(bridge_forecast_name("gru"), config)
        model.load_pretrained_history(source)
        zscores = ZScores(1, 2, 3, 4, 5, 6, 4, 2, 10, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecast.pt"
            save_checkpoint(
                path,
                model,
                config,
                zscores,
                {"training_data": "prepared.csv"},
                1,
                0.5,
                bridge_forecast_name("gru"),
                pretraining={"weights_loaded": True},
            )

            loaded, loaded_config, loaded_zscores, checkpoint = load_checkpoint(
                path, torch.device("cpu")
            )

        self.assertEqual(loaded_config, config)
        self.assertEqual(loaded_zscores, zscores)
        self.assertTrue(checkpoint["pretraining"]["weights_loaded"])
        self.assertEqual(set(loaded.state_dict()), set(model.state_dict()))

    def test_linear_and_persistence_evaluation_helpers(self):
        indoor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        forecast = torch.tensor(
            [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]]
        )
        target = 1 + 2 * indoor[:, None] + 3 * forecast
        batches = [
            {
                "history": torch.stack(
                    (
                        torch.zeros(4, 2),
                        indoor[:, None].expand(4, 2),
                    ),
                    dim=2,
                ),
                "forecast": forecast[..., None],
                "target": target,
            }
        ]

        coefficients = fit_linear_baseline(batches)
        prediction = linear_prediction(indoor.double(), forecast.double(), coefficients)
        report = metric_report(prediction, target.double(), [1, 2])

        torch.testing.assert_close(prediction, target.double())
        self.assertAlmostEqual(report["by_horizon"]["2"]["rmse"], 0.0, places=10)

    def test_one_epoch_transfer_training_uses_prepared_csv(self):
        config = _config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bridge_path = root / "bridge.pt"
            training_data = root / "training.csv"
            torch.save(_checkpoint("single-self-attention-encoder", config), bridge_path)
            _write_singular_training_data(training_data, config)
            args = build_parser().parse_args(
                [
                    "train",
                    "--pretrained-checkpoint",
                    str(bridge_path),
                    "--training-data",
                    str(training_data),
                    "--history-hours",
                    "12",
                    "--prediction-hours",
                    "6",
                    "--forecast-embedding-dim",
                    "8",
                    "--forecast-heads",
                    "2",
                    "--forecast-head-dim",
                    "4",
                    "--forecast-layers",
                    "1",
                    "--forecast-feedforward-dim",
                    "16",
                    "--forecast-dropout",
                    "0",
                    "--decoder-embedding-dim",
                    "8",
                    "--decoder-heads",
                    "2",
                    "--decoder-head-dim",
                    "4",
                    "--decoder-layers",
                    "1",
                    "--decoder-feedforward-dim",
                    "16",
                    "--decoder-dropout",
                    "0",
                    "--epochs",
                    "1",
                    "--freeze-history-epochs",
                    "1",
                    "--batch-size",
                    "1",
                    "--device",
                    "cpu",
                    "--checkpoint",
                    str(root / "inference" / "checkpoints" / "training_bridge.pt"),
                    "--metrics-output",
                    str(root / "inference" / "metrics" / "training_bridge.csv"),
                    "--loss-plot",
                    str(root / "inference" / "graphs" / "training_bridge.png"),
                    "--report-output",
                    str(root / "inference" / "reports" / "training_bridge.csv"),
                ]
            )
            train(args)

            checkpoint = torch.load(
                root / "inference" / "checkpoints" / "training_bridge.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertTrue(
                (root / "inference" / "metrics" / "training_bridge.csv").is_file()
            )
            self.assertTrue(
                (root / "inference" / "graphs" / "training_bridge.png").is_file()
            )
            with (
                root / "inference" / "reports" / "training_bridge.csv"
            ).open(encoding="utf-8", newline="") as source:
                report = list(csv.DictReader(source))
            self.assertEqual(len(report), 6)
            self.assertEqual(
                {(row["split"], row["horizon_hours"]) for row in report},
                {
                    (split, horizon)
                    for split in ("validation", "temporal_test", "location_test")
                    for horizon in ("3", "6")
                },
            )

        self.assertEqual(
            checkpoint["model_name"],
            bridge_forecast_name("single-self-attention-encoder"),
        )
        self.assertTrue(checkpoint["pretraining"]["weights_loaded"])
        self.assertEqual(checkpoint["training_config"]["forecast_horizons"], [6])

    def test_masked_report_contains_reconstruction_and_bridge_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            training_data = root / "training.csv"
            checkpoint.write_bytes(b"checkpoint")
            training_data.write_text("training", encoding="utf-8")
            metadata = {
                "model_name": "dual-encoder-cross-fusion",
                "completed_stages": ["points", "tempo_bridge_50"],
                "stage": "tempo_bridge_50",
                "validation_metrics": {
                    "loss": 0.2,
                    "indoor_rmse": 2.5,
                    "outdoor_rmse": 3.5,
                    "target_count": 200,
                },
                "training_data_sha256": "data-hash",
            }
            metrics = [
                {
                    "global_epoch": 1,
                    "stage": "points",
                    "stage_epoch": 1,
                    "train_loss": 0.3,
                    "validation_loss": 0.25,
                    "validation_indoor_rmse": 2.4,
                    "validation_outdoor_rmse": 3.4,
                    "train_target_count": 100,
                    "validation_target_count": 100,
                    "improved_checkpoint": True,
                    "pipeline_elapsed_seconds": 1,
                },
                {
                    "global_epoch": 2,
                    "stage": "tempo_bridge_50",
                    "stage_epoch": 1,
                    "train_loss": 0.25,
                    "validation_loss": 0.2,
                    "validation_indoor_rmse": 2.5,
                    "validation_outdoor_rmse": 3.5,
                    "train_target_count": 200,
                    "validation_target_count": 200,
                    "improved_checkpoint": True,
                    "pipeline_elapsed_seconds": 2,
                },
            ]
            output = root / "report.csv"

            rows = write_masked_final_report(
                output, checkpoint, training_data, metadata, metrics
            )

            self.assertTrue(output.is_file())
            self.assertEqual(
                [row["stage_type"] for row in rows], ["reconstruction", "bridge"]
            )
            self.assertTrue(rows[-1]["is_final_stage"])
            self.assertEqual(rows[-1]["indoor_rmse_ug_m3"], 2.5)


def _write_singular_training_data(
    path: Path, config: BridgeForecastConfig
) -> None:
    metadata = (
        "sample_index",
        "split",
        "location_id",
        "sensor_id",
        "model_name",
        "history_hours",
        "prediction_hours",
        "anchor_time_utc",
    )
    history = tuple(
        f"history_{hour:03d}_{feature}"
        for hour in range(config.history_hours)
        for feature in (
            "tempo_pm25_ug_m3",
            "indoor_pm25_ug_m3",
            "daily_sin",
            "daily_cos",
            "weekly_sin",
            "weekly_cos",
            "annual_sin",
            "annual_cos",
        )
    )
    forecast = tuple(
        f"forecast_{hour:03d}_{feature}"
        for hour in range(1, config.prediction_hours + 1)
        for feature in (
            "naqfc_pm25_ug_m3",
            "daily_sin",
            "daily_cos",
            "weekly_sin",
            "weekly_cos",
            "annual_sin",
            "annual_cos",
        )
    )
    target = tuple(
        f"target_{hour:03d}_indoor_pm25_ug_m3"
        for hour in range(1, config.prediction_hours + 1)
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(metadata + history + forecast + target)
        for index, split in enumerate(
            ("train", "validation", "temporal_test", "location_test")
        ):
            history_values = [
                value
                for hour in range(config.history_hours)
                for value in (
                    float("nan") if hour % 3 else 8 + hour,
                    2 + index + hour / 10,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                )
            ]
            forecast_values = [
                value
                for hour in range(config.prediction_hours)
                for value in (5 + hour, 0, 1, 0, 1, 0, 1)
            ]
            writer.writerow(
                (
                    index,
                    split,
                    f"location_{index}",
                    index + 1,
                    "transformer-cyclical",
                    config.history_hours,
                    config.prediction_hours,
                    f"2024-01-0{index + 1}T00:00:00Z",
                    *history_values,
                    *forecast_values,
                    *(3 + index + hour / 10 for hour in range(config.prediction_hours)),
                )
            )


if __name__ == "__main__":
    unittest.main()
