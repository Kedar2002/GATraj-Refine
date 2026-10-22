# GATraj-Refine

A small endpoint-and-smooth-path residual refiner around an **original, frozen
GATraj**. No improved benchmark accuracy is claimed. This is an experimental
adaptation of established coarse-to-fine/residual refinement ideas.

[Open in Colab](https://colab.research.google.com/github/Kedar2002/GATraj-Refine/blob/main/Colab_Refine.ipynb)

## What changes

The GATraj temporal encoder, interaction graph, original attention computation,
20-mode decoder and original backbone training loss are retained. There is no
aWTA loss, RiskMotion feature module or attention correction. The refiner uses
observed displacements, each original predicted trajectory and frozen social
context to predict a bounded endpoint correction and three smooth path controls.
The defaults add **7,240 parameters** to the 293,131-parameter backbone.

Backbone parameters are frozen and stay in eval mode, including dropout, during
refiner training. The original output scales and mode logits are retained.
Probability calibration after moving means is not established by ADE/FDE.

Refinement starts with zero corrections. Its off switch bypasses the extra
network and returns the same frozen original outputs. These are correctness
properties, not a guarantee that learned corrections improve unseen scenes.

## Colab workflow

Use `Colab_Refine.ipynb` with a GPU runtime. It follows the previous RiskMotion
and aWTA workflow: mount Drive, clone/install, run CPU checks, configure one
fold, automatically resume training, and evaluate paired fixed checkpoints.

**Set `BASE_CHECKPOINT` to your actual original GATraj checkpoint.** For example,
`/content/drive/MyDrive/GATraj/savedata/1/1/GATraj/GATraj_1000.tar` may be correct
for a previously nested Hotel backup. The suggested notebook path may differ
from your Drive. An unreadable legacy checkpoint must first be recovered;
changing this code cannot repair an already truncated file.

The default `TRAIN_BACKBONE=False` reuses your original weights and trains only
the refiner. `TRAIN_BACKBONE=True` first trains a fresh original backbone under
a separate 1,000-epoch schedule. It is useful if no compatible baseline exists
or if a validation scene needs to be excluded from backbone training. It incurs
the full original backbone training cost.

- Fold IDs: 0 ETH, 1 Hotel, 2 Zara1, 3 Zara2, 4 University.
- Refiner defaults: 100 epochs, batch size 32, learning rate 1e-4, cosine eta_min
  1e-5, clip 1, random rotations. This is an unvalidated starting budget.
- Set `REFINER_STOP_AT_EPOCH=1` for startup, keeping `REFINER_EPOCHS=100` fixed.
- Every completed epoch is saved directly to Drive using a temporary file before
  replacement. After successful chunks the notebook optionally keeps every
  tenth epoch and the latest. Keep the original source checkpoint in Drive.
- Refiner checkpoints include the frozen backbone, refiner, optimizer, scheduler,
  RNG and original-source SHA256. They never overwrite the original checkpoint.
- Refiner checkpoints live under
  `MyDrive/GATraj-Refine/runs/refiners/<fold>/<experiment>/`.
- Fresh optional backbones live under
  `MyDrive/GATraj-Refine/runs/backbones/<fold>/<experiment>/`.
- Simultaneous sessions must not write to the same experiment directory.

## Similar training command

```bash
python train.py --stage refiner --base_checkpoint /path/to/original_GATraj_1000.tar \
  --test_set 1 --seed 1 --num_epochs 100 --epochs_per_run 10 \
  --x_encoder_layers 3 --eta_min 1e-5 --batch_size 32 --learning_rate 1e-4 \
  --randomRotate True --final_mode 20 --neighbor_thred 10 --using_cuda True \
  --gpu 0 --clip 1 --pass_time 2 --ifGaussian False --SR True --input_offset True \
  --endpoint_refine True --shape_refine True --save_base_dir ./runs
```

Resume with `--load_model <last-refiner-epoch>` and the same full refiner horizon
and arguments. Refiner epochs are separate from the original checkpoint's
training epoch. A refiner epoch 10 may contain the original epoch-1000 backbone.

A saved refiner checkpoint is self-contained. If its original source file has
moved, direct CLI resume can pass the recorded `--base_sha256` instead of a
readable source file. The notebook keeps and checks the source file to resolve
its experiment and verify initialization at every setup.

To train an original backbone here, use `--stage backbone --num_epochs 1000
--learning_rate 5e-4` with the other original settings and no base checkpoint.
Checkpoint/configuration YAML files are records; CLI arguments are authoritative.
Architecture/experiment mismatches on resume cause an explicit error.

## Switches and paired evaluation

| Setting | Effect |
|---|---|
| `--endpoint_refine True --shape_refine True` | Train both corrections |
| `--endpoint_refine True --shape_refine False` | Endpoint-only ablation |
| `--endpoint_refine False --shape_refine True` | Path-only ablation |
| `--refiner False` during evaluation | Original frozen-backbone bypass |

Each component configuration has its own run directory. Path-only refinement
has zero terminal correction, so its FDE must remain the same as the frozen
baseline; it is an ADE-focused ablation, not a way to improve FDE alone.

Run `evaluate.py` using the same settings plus `--phase test --load_model 100`.
Evaluate once with `--refiner False` and once with `--refiner True`; reports are
separate `evaluation_100_baseline.json` and `evaluation_100_refined.json` files.
Both use the exact same embedded original checkpoint and K=20. `compare.py`
accepts these paired reports and requires the predetermined final refiner epoch.

## Correction and objective

For u=t/T, add `u*delta_goal + sum_j B_j(u)*control_j` to each original mean.
The shape basis consists of the interior Bernstein polynomials of degree J+1,
so it vanishes at both observation and prediction endpoints. Positive basis
weights sum to at most one. Endpoint and control coefficients are bounded with
tanh at 1.0 m and 0.5 m per coordinate by default. The last network projection
is zero-initialized with a live gradient path.

The shared MLP receives 14 observed displacement coordinates, 24 original path
coordinates and 64 frozen context features, then uses 64 hidden units and an
8-coordinate correction head. There are no cross-agent mode-index assumptions.

Refiner training retains the original backbone's per-agent winning mode
assignment and applies its original Laplace (or optional Gaussian) regression
loss to the corresponding corrected mean with the original scale. It adds a
SmoothL1 endpoint term (default weight 0.1) and mean-squared correction penalty
(default weight 0.01). The frozen classifier term is retained as a constant
part of the reported loss. These refiner settings are hypotheses, not tuned
benchmark values. Original backbone training does not use these added terms.

Training predicts proposals online from the frozen backbone, so augmentation
and observations remain consistent. No future labels enter predictions or
refinement at inference. No train/test-fitted trajectory basis or extra data
are required; the smooth basis is fixed analytically.

## Validation and reporting

Default `--validation_set -1` trains/refines on four scenes and evaluates the
held-out fifth scene. For a separate pilot, choose a validation scene distinct
from the outer test scene. Both are excluded from backbone/refiner training;
only the validation scene is evaluated. Such pilot scores are not the standard
four-scene benchmark. A validation pilot requires source metadata confirming
that this validation scene was excluded from backbone training; legacy sources
cannot establish that provenance.

Legacy original checkpoints without metadata can initialize a standard final
experiment, but the operator must manually confirm fold and architecture.
Known aWTA, RiskMotion, corrected-attention or wrong-fold sources are rejected.
A baseline arm from GATraj-aWTA is acceptable when its metadata confirms no
modification. Original-fold metadata cannot be inferred from a filename alone.

The validation comparison retains baseline if either ADE or FDE worsens; it
is not a test guarantee. Test comparisons report both variants without choosing
a winner using test results. Use validation to decide endpoint/path ablations,
regularization and training budget, then keep settings and final epoch fixed.
A previously trained four-scene original can be reused for the final experiment,
but not to create an untouched validation experiment on one of its training
scenes. In-sample backbone forecasts can underestimate held-out errors, and
a learned refiner can still overfit or reduce diversity.

Report minADE20/minFDE20, all five folds and paired seeds. Average folds equally.
Distinguish refiner-seed variation with one fixed backbone from variation across
independently trained backbones. For latency comparisons use the same GPU and
report refiner overhead. Small roundoff differences can occur between PyTorch
inference kernel paths with different gradient flags; the exact identity checks
compare the same frozen-backbone execution path.

## Verification and attribution

See [VERIFICATION.md](docs/VERIFICATION.md) for actual executed checks. Run
`python -m unittest discover -s tests -v` to repeat the CPU checks. Colab runs
them before GPU training. Full convergence and GPU improvement are not verified.

The original model is GATraj (Liu et al., ISPRS JPRS 2023), derived from
[Kedar2002/GATraj](https://github.com/Kedar2002/GATraj) commit `08a0042`, with
common runtime/cache/validation handling adapted from the preceding experiment.
The original MIT license is retained.
[Original authors' code](https://github.com/mengmengliu1998/GATraj).

Conceptual precedents, not implementations vendored here:
[Rethinking CNNs for trajectory refinement](https://doi.org/10.1016/j.patcog.2024.110883),
[EigenTrajectory](https://github.com/InhwanBae/EigenTrajectory),
[TrajFine](https://openaccess.thecvf.com/content/CVPR2024W/WAD/html/Wang_TrajFine_Predicted_Trajectory_Refinement_for_Pedestrian_Trajectory_Forecasting_CVPRW_2024_paper.html).
Residual learning, endpoint conditioning and coarse-to-fine refinement are
existing ideas. This repository does not claim a new algorithm or improved
accuracy without experimental evidence. It implements ETH/UCY, not nuScenes.
