# Masked Pretraining for Indoor PM2.5 Forecasting

## Core idea

Pretrain the history encoder by reconstructing deliberately hidden indoor and outdoor PM2.5 observations. This is **masked pretraining**; patch geometry stays fixed.

Natural missingness and artificial masking remain separate in the training
targets. Both enter the model as `-9` in the affected normalized PM2.5 slot,
but naturally missing PurpleAir values have no target. Artificial masks apply
only to known values, and loss is calculated only on them. The model input
remains the final eight features: outdoor PM2.5, indoor PM2.5, and six cyclical
time features. TEMPO is reserved for later forecasting work.

## Static training assignments

Training consumes only
`inputs/masked_pretraining/exclusion_aware/training_data.csv`. Exclusion-aware
matching resolves all outdoor fallbacks before generation, and each interval
row includes its indoor and outdoor PM2.5 readings. One time series and one
train/validation split unit are built per indoor sensor. A 168-hour window may
cross adjacent outdoor sensor assignments, but it cannot cross an interval gap
or contain two indoor sensor IDs.

## Training regime

Advance when validation improvement plateaus:

1. Hide 10-15% of isolated hours to learn continuity.
2. Hide 20-25% in 2-3 hour blocks to learn event shapes.
3. Hide 30-40% in mixed 1-, 3-, and 6-hour blocks.
4. Hide indoor blocks with outdoor history visible to teach cross-channel responses.
5. Hide the final 3, 6, and 12 indoor hours to learn extrapolation.

Retain earlier mask types in later stages.

## Transition to forecasting

After suffix reconstruction succeeds, retain the history encoder and replace the reconstruction head with the future-outdoor encoder and indoor forecast decoder. Briefly freeze the history encoder, then fine-tune the complete model. Expand forecasting from 3 to 6, 12, 24, and 36 hours.

Compare against persistence, a linear baseline, and an equal-budget model trained without masked pretraining.
