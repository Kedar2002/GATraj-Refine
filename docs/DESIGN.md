# Next candidate: endpoint-and-shape residual refinement

Status: implemented here as an experimental refiner. No improved benchmark performance is established.

## Why this candidate

Freeze a trained original GATraj and learn a small residual refiner over its
20 trajectory means. This preserves the original trained model and avoids
changing its interaction graph, temporal encoder, mode-assignment mechanism
or training objective while training that backbone. The added stage corrects
outputs rather than relearning all motion representations.

A disabled refiner must return exactly the baseline. Zero-initializing its
last projection also starts enabled refinement at exactly the baseline.
Neither property guarantees a trained refiner will generalize better.

## Proposed mechanism

For each agent and mode, consume observed displacements, the original mode's
predicted trajectory, and optionally its frozen encoded history. A small
shared temporal CNN/MLP predicts an endpoint correction and a few smooth
intermediate-path correction coefficients:

    refined[k,t] = baseline[k,t] + (t/T)*delta_goal[k]
                   + sum_j basis[j,t]*coefficient[k,j]

Choose fixed basis functions that vanish at t=0 and t=T, so the endpoint
correction and intermediate shape correction have separate roles. All
corrections vanish at the observation boundary. Output exactly 20 trajectories.
No true future positions may enter the refiner at inference.

Train only the refiner first, with the original backbone frozen and in eval
mode. Retain the original backbone loss; during refiner training supervise
corrections using training labels, regularize correction magnitude, and avoid
forcing all 20 modes toward the same ground truth. Begin with the baseline's
winning mode assignment to preserve its learned alternatives. This assignment
is a proposed implementation detail, not an established optimum.

A zero-initialized correction layer should have a nonzero-gradient path; do
not multiply two branches both initialized to zero. Keep baseline and refiner
checkpoints separate. Fixed baseline probabilities/scales are not guaranteed
to remain calibrated after changing trajectory means; report displacement
metrics separately from probability calibration.

## Training and evaluation controls

1. Confirm the original checkpoint loads and reproduce its stored metric first.
2. Check bitwise equality with refinement off and with zero-initialized output.
3. Use a validation scene excluded from BOTH backbone and refiner training
   for method/schedule selection. A frozen backbone that already trained on
   that scene does not provide an untouched validation experiment.
4. A trained four-scene baseline can be reused for final fixed-setting
   refinement experiments, but its in-sample forecasts may underestimate
   held-out error. This mismatch is a risk, not a reason to use test labels.
5. Cache training predictions only if observations/augmentation match the
   intended refiner training inputs. No test predictions or labels in fitting
   a trajectory basis, anchors, feature normalization or refiner parameters.
6. Compare original, endpoint-only, shape-only, and endpoint+shape variants
   on validation before full five-fold experiments. Select a checkpoint and
   settings before looking at test metrics. Keep K=20.
7. A short pilot can reveal unstable training or lack of a validation signal;
   it cannot establish that a different run will never improve. A practical
   stopping decision should be documented rather than repeatedly tuning tests.

This adds a refiner-training stage. The original backbone need not be retrained
for the final fixed-setting experiment when a valid baseline checkpoint is
available. Compute savings and an appropriate refiner epoch budget need to be
measured, not promised as a fixed fraction of 1,000 epochs.

## Relevant existing mechanisms

- Yoon et al., Pattern Recognition 2025, Rethinking convolutional neural
  networks for trajectory refinement. Uses a coarse-to-fine CNN and expected
  goals from pretrained PECNet/V2-Net on ETH/UCY. This supports the refinement
  direction, not this exact GATraj residual design. A verified official code
  release was not located in this search.
  https://doi.org/10.1016/j.patcog.2024.110883
- EigenTrajectory, ICCV 2023: compact trajectory descriptors and anchor-based
  refinement, with official code supporting ten backbones. GATraj is not in
  its supported-backbone list. Our optional compact correction would be an
  adaptation, not a drop-in official GATraj integration.
  https://github.com/InhwanBae/EigenTrajectory
- TrajFine, CVPR Workshops 2024: predicted future interaction refinement using
  diffusion and scene mixup. Relevant evidence for future-trajectory refinement,
  but its full diffusion pipeline is a larger engineering/training change.
  https://openaccess.thecvf.com/content/CVPR2024W/WAD/html/Wang_TrajFine_Predicted_Trajectory_Refinement_for_Pedestrian_Trajectory_Forecasting_CVPRW_2024_paper.html
- GATraj+PnS, ICCV Workshops 2023: direct published GATraj improvements are
  small: ETH 0.26/0.42 to 0.26/0.40, Hotel unchanged at 0.10/0.15, University
  0.21/0.38 to 0.21/0.37, Zara1 unchanged at 0.16/0.28, Zara2 0.12/0.21
  to 0.11/0.20. A verified official PnS implementation was not located in
  this search. Reimplementation is an alternative with direct backbone
  evidence, but no guarantee of larger gains or a new contribution.
  https://openaccess.thecvf.com/content/ICCV2023W/ROAD%2B%2B/papers/Liu_Tracing_the_Influence_of_Predecessors_on_Trajectory_Prediction_ICCVW_2023_paper.pdf

Residual learning, endpoint conditioning and trajectory refinement are existing
ideas. This design must be described as an adaptation unless a distinct new
mechanism and its benefit are established through literature review and tests.
