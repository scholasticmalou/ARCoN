# ARCoN: Adaptive Response Coordination Networks for Neural Garment Simulation

Research code accompanying ARCoN, with training, inference, evaluation, and the final 25k checkpoint.

## Contents

- `train.py`: initialize the model, run the documented training course, or resume a full training checkpoint.
- `infer.py`: complete autoregressive prediction with the supplied 25k weights or a trained material/final-stage ARCoN checkpoint.
- `measure.py`, `score.py`: six direct material/body/self-contact measurements and penalty-adjusted JAR/JCV, including the four-resolution extensions.
- `verify_scores.py`, `examples/`: 1,160 frozen score comparisons from complete saved measurements; these do not contain licensed body/garment assets.
- `arcon/models/`: geometry-conditioned graph updates, spatial interactions, material/body relation responses and self-supervision.
- `arcon/control/`: the original physical training loops, staged construction and native evaluation adapters, with public module names.
- `arcon/compat.py`: compatibility with inherited loader namespaces and historical checkpoint metadata.
- `checkpoints/arcon_25000.pth`: the reported final inference weights, including normalization state.
- `configs/`: resolved training settings, stage boundaries, fixed quality scales and baseline settings.
- `assets/calibration.json`: the frozen TRAIN self-contact calibration used by the paper training course.
- `external/`: attributed Pb4U/HOOD infrastructure and the ContourCraft, SENC and MCO baseline source snapshots.

The ARCoN implementation has its own module and class names. Inherited `model.pb4u`, runner, dataset and criterion keys are upstream API/checkpoint namespaces; they are retained in the compatibility layer rather than changing parameter keys or removing attribution.

## Environment

Linux and an NVIDIA CUDA GPU are used for model execution. `tested_environment.json` records the actual environment used to collect the reported checkpoint. The frozen metric check requires only Python and NumPy:

```bash
python verify_scores.py
```

For model execution, install compatible builds of PyTorch, PyTorch Geometric, `torch_scatter`, PyTorch3D and SMPL-X, together with the Python dependencies in `requirements.txt`. Follow the retained upstream environment instructions for the native dataset and geometric libraries. CUDA extensions must be built against the installed PyTorch/CUDA pair; the historical upstream `environment.yml` is retained as an upstream reference, not presented as the tested environment of this package.

For direct body-distance measurements, install NVIDIA Warp. Self-intersection measurements use ContourCraft's native `cccollisions` detector at its original zero threshold and 320-candidate setting. Training additionally needs the CCCollisions headers and CUDA sample headers used by the counted self-intersection objective. The small PyTorch compatibility patch is in `dependency_patches/`. Apply it from the CCCollisions checkout with `patch -p1 < /path/to/ARCoN/dependency_patches/cccollisions_torch211.patch` when building against the recorded PyTorch version. These third-party dependencies are obtained separately under their own terms; see `THIRD_PARTY_NOTICES.md`.

## Data preparation

The package does not redistribute SMPL model files, motion/garment datasets, or baseline checkpoints. Obtain the native assets through the data preparation instructions in `external/PB4U-GNet-main/README.md` and accept their original access/licensing terms.

The `--data-root` argument is the Pb4U-format data directory containing:

```text
data-root/
  aux_data/
    smpl/SMPL_FEMALE.pkl
    smpl_aux.pkl
    garments_dict.pkl
    datasplits/train.csv
  vto_dataset/smpl_parameters/
```

Inference accepts an explicit native motion `.pkl`, an explicit garment dictionary `.pkl`, and the key of the garment within that dictionary. A garment entry retains its native rest vertices, faces, skinning weights, fixed-vertex information and hierarchy metadata. A motion file retains native SMPL pose/translation/shape arrays. Use the existing upstream conversion to create these files; do not substitute rendered images for physical inputs. The paper's four held-out motions are `07_02` (walking), `01_01` (jumping), `104_54` (running) and `55_27` (arm movements). Their complete prediction horizons are 112, 270, 92 and 240 respectively. Evaluation preserves the native startup interpolation and correspondence. Training warmup checkpoints use their phase-specific settings; `infer.py --checkpoint` accepts the material/final-stage model checkpoints.

## Inference

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --data-root /path/to/data-root \
  --external-root /path/to/dependency-workspace \
  --sequence /path/to/native/07_02.pkl \
  --garment-dict /path/to/prepared/Lv.2_tshirt.pkl \
  --garment tshirt_KEY_FROM_DICTIONARY \
  --output outputs/walking_tshirt.npz
```

The output contains `pred`, `obstacle`, `cloth_faces`, `obstacle_faces`, `rest_pos` and `vertex_type`, in metres. The model uses one unchanged checkpoint across mesh levels and the original physical-distance propagation adaptation. Omit `--steps` for complete-trajectory evaluation. A positive `--steps` value is a diagnostic prefix, not a complete-trajectory result.

## Direct measurements and joint quality

```bash
CUDA_VISIBLE_DEVICES=0 python measure.py \
  --rollout outputs/walking_tshirt.npz --output outputs/walking_tshirt_metrics
python score.py --metrics outputs/walking_tshirt_metrics
```

If Warp and the native collision detector use separate environments, run `measure.py --part geometry` in the model/Warp environment and `measure.py --part self` in the collision-detector environment, with the same input/output paths. The two commands write distinct files without replacing one another. This is also how the original evaluation pipeline separates its native namespaces.

The six measurements are body-penetration rest-area fraction, whole-garment area-weighted penetration depth (metres), area-weighted Green-strain RMS, edge-length relative-change RMS, self-intersection rest-area fraction, and rest-area fraction affected by at least ten consecutive intersecting endpoints. Area fractions are 0--1, not percentages. Measurement formulas and the native detector are retained from the paper pipeline. Whole-cloth direct contact measurements are distinct from the free-vertex-only native Collision objective reported in physical-objective tables.

JAR/JCV aggregate all recorded endpoints of a trajectory before zero-clipping the penalty-adjusted score. Within each material/body/self group, the most severe relative boundary exceedance determines its penalty; the group is not counted twice. The main requirement domain is `[0.5,2]`. `configs/evaluation.json` gives the frozen scales, group weights, serious-violation boundaries and position-consistency scale.

For four-resolution evaluation, pass four corresponding metric directories and one endpoint-aligned position-consistency vector in metres:

```bash
python score.py --metrics outputs/lv1 outputs/lv2 outputs/lv3 outputs/lv4 \
  --position-consistency outputs/position_consistency_rms.npy
```

The scalar position-consistency vector uses the paper's fixed rest-surface correspondence, not a correspondence fitted to predicted positions. Frozen complete-sequence examples, including this vector, are supplied for independent score verification.

## Training from initialization

No pretrained model checkpoint is required to start training. The public entry constructs seed-57 parameters and fresh optimizer/scheduler/normalizer state, then runs the original physical training loops through the paper's absolute iteration boundaries:

| Data iterations | Training setting |
|---|---|
| 0--12,500 | Initial graph-network course with the fixed direction update and fine-scale residual |
| 12,500--15,000 | Surface-supported body-contact readout |
| 15,000--20,000 | Continued surface-contact course, retaining the original loader restart at 17,500 |
| 20,000--22,500 | Proposal-conditioned contact and material-geometry-conditioned bending response |
| 22,500--25,000 | Final continuation of the same model |

`configs/training_course.json` specifies the boundaries and loader restart. The course uses the original 5,000-iteration rollout curriculum, up to five physical steps per sampled window, and the supplied fixed TRAIN calibration. The first 99 data iterations accumulate normalizer statistics before the first optimizer update. The self-contact loss activates at iteration 8,000 and ramps over 4,000 iterations; it is not recalibrated on evaluation data. The original gradient-based calibration routine remains in `arcon.control.initialization.calibrate` for inspecting how the supplied scale was obtained.

Prepare the authorized training data and a dependency workspace containing:

```text
external-root/external/CCCollisions-torch211-compat/{src,include}
external-root/external/cuda-samples-12.4/Common/
```

Run the complete course on one visible GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data-root /path/to/data-root \
  --external-root /path/to/dependency-workspace \
  --output outputs/train_arcon
```

The driver creates its own intermediate checkpoints. They contain model/normalizer tensors, optimizer, scheduler and random states. The original three within-step rounds are differentiated jointly; predicted states are detached between physical steps. The resolved configuration and loss/update records are saved with each stage.

To check the entry on real training data without running the complete course:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data-root /path/to/data-root \
  --external-root /path/to/dependency-workspace \
  --output outputs/train_entry_check --until-step 102
```

This includes the normalizer warmup and three actual parameter updates. It is a short training check, not a replacement for the complete course. Print the plan without CUDA dependencies using `python train.py --print-plan`.

Resume any full checkpoint into a new output directory; `--until-step` is an absolute cumulative iteration:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data-root /path/to/data-root \
  --external-root /path/to/dependency-workspace \
  --resume /path/to/your/full_training_checkpoint.pth \
  --output outputs/train_resumed --until-step 25000
```

Each original stage/restart restores the saved RNG state and starts a new native loader pass; the historical checkpoints do not store a loader cursor. The trained checkpoint path is recorded in `training_result.json`. The distributed 25k file is inference-only: it intentionally excludes optimizer/scheduler/RNG state and is not a resume checkpoint. Its 25,000 data iterations correspond to 75,004 physical-step forwards and 74,905 optimizer updates. Intermediate training checkpoints remain internal to the author's training runs and are not needed by a reader starting from initialization.

## Baseline source and evaluation settings

`configs/baselines.json` records the six comparison implementations, checkpoint filenames, native settings and construction adapters. Source code is supplied under:

```text
external/PB4U-GNet-main/     # Pb4U plus its released HOOD and Fine15 implementations
external/ContourCraft-main/
external/SENC/
external/MCO/
```

Obtain author weights and licensed data through the corresponding upstream README files. Place optional downloaded weights under `checkpoints/baselines/`; none are bundled or required for ARCoN training. Fine15 is the MGN-style implementation released with Pb4U, not an assertion that its checkpoint is the original MeshGraphNets paper model.

The native adapters are `arcon.control.baseline_runtime` (Pb4U/HOOD/Fine15/ContourCraft), `arcon.control.author_baselines` (SENC), and `arcon.control.mco_baseline` (MCO). They retain the original model/runner and input preparation. MCO uses the listed frictionless-trained predictor, complete contact operator, profile and friction coefficient, rather than replacing it with the standard ContourCraft checkpoint. Its locations are configurable with `ARCON_MCO_SOURCE`, `ARCON_MCO_CHECKPOINT`, `ARCON_MCO_DATA`, `ARCON_MCO_MOTION_ROOT` and `ARCON_MCO_ROOT` (runtime/cache output).

For a common comparison, keep the same rest mesh, motion inputs, material parameters and endpoint axis across methods. The adapters and `arcon.control.trajectory_evaluation` retain the paper's common-input binding; do not substitute remeshed baseline inputs or reset the garment from simulated ground truth. Once complete rollouts are exported in the six-array format documented above, apply the same `measure.py` and `score.py` commands to every method. The frozen score fixtures already include all six baselines and ARCoN. Baseline inference was not rerun as part of this source-layout cleanup.

## Validation and attribution

`VALIDATION.json` separates the original numerical validation from the public source-layout/training checks. The release checks include 1,160 saved-score comparisons; six-measurement recomputation on 112 endpoints; complete inference replays; from-initialization normalizer warmup and optimizer updates; full-state save/resume; and real training windows at the original stage handoffs. A complete 25k retraining is not part of the packaging check. CUDA autoregressive replays are not bitwise identical; measured replay differences are recorded rather than described as exact geometry reproduction.

All published inference tensors remain unchanged. Upstream code, licenses and attribution stay in `external/`, `licenses/` and `THIRD_PARTY_NOTICES.md`. ARCoN-specific code is distributed under the repository's MIT license. Third-party components retain their original licenses and attribution; separately obtained datasets, body models and geometric dependencies are governed by their upstream terms.
