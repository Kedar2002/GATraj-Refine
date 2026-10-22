# Verification record

Executed locally on 22 October 2026 with Windows, Python 3.10 and CPU PyTorch
2.6.0. These checks do not establish improved ADE/FDE or Colab GPU performance.

## Eight focused CPU checks passed

- Zero-initialized means/scales/logits equal the loaded original under the same
  frozen inference path; bypass remains exact even after changing refiner weights.
- Endpoint-only correction follows the linear ramp; shape-only correction is
  zero at the endpoint; basis values vanish at both boundaries and obey bounds.
- Backbone remains in eval mode during training, receives no gradients, and
  its complete state dictionary stays unchanged after an optimizer step.
- Refiner receives finite, nonzero gradients from the supervised objective.
- Predictions with nonzero corrections do not change when future labels change.
- Wrong-fold, changed source hash and invalid validation provenance are rejected.
- A complete GATraj+refiner model/Adam/scheduler/RNG continuation matches an
  uninterrupted run exactly on CPU. The original source is deleted before
  resuming in this check to verify that the new checkpoint is self-contained.
- Experiment names and CLI resolution agree, evaluation bypass addresses the
  same checkpoint, components/source hashes separate runs, and report comparison
  selects only on validation, with fallback on an ADE/FDE tradeoff.

The above bullets summarize eight unittest cases; some cases contain several
related assertions. Refiner input dimensions and 20-mode output shapes are
exercised through the full wrapper.

## Actual original checkpoint and data startup

Loaded the supplied original Hotel-fold epoch-1000 checkpoint from the existing
base GATraj repository. Its legacy serialization loaded successfully in this
local runtime; it lacks experiment metadata, so its provenance warning was
shown rather than pretending the fold could be machine-verified.

Trained one refiner epoch using the original real ETH/UCY loader, saved a ZIP
checkpoint, resumed in a new process for epoch 2, and ran paired bypass/refined
evaluation successfully. The refiner cosine horizon stayed at 100 epochs with
`epochs_per_run=1`. Training had 96 batches; Hotel evaluation had 13 batches
and 1,075 agent-windows. The backbone remained frozen in both epochs.

Default parameter counts: original 293,131; refiner 7,240; total 300,371.
Reports distinguish total parameters from active parameters in bypass mode.
No two-epoch scores are published as a trained improvement and no settings
were selected from those test scores. Full refiner convergence, multiple seeds,
CUDA behavior and Drive interruption behavior remain unverified.

The notebook JSON, its code cells and embedded CPU-probe source were checked
for syntax. Its actual source-checkpoint probe also executed successfully on
CPU with the original Hotel checkpoint. Actual Colab execution has not been
performed here. Its setup
repeats the CPU suite in the user's runtime, then validates the actual chosen
source checkpoint and initialization equality before training.
