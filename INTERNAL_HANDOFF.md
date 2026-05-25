# INTERNAL HANDOFF DOCUMENT
## Orthographic Gaussian Splatting for Volumetric Anatomical Visualization
### Full Research Archive — Technical Transfer Brief

**Researcher:** Mushfiq  
**Thesis Title:** Orthographic Gaussian Splatting for Volumetric Anatomical Visualization  
**Archive Root:** `C:\Personal_Endeavours\Thesis\`  
**Document Written:** 2026-05-25  
**Intended Audience:** Incoming research/engineering team with zero prior context

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Research Progression — Three Phases](#3-research-progression--three-phases)
4. [Architecture Overview](#4-architecture-overview)
5. [Technical Components — Deep Dive](#5-technical-components--deep-dive)
6. [Experimental Pipeline](#6-experimental-pipeline)
7. [Research Findings](#7-research-findings)
8. [Engineering Challenges](#8-engineering-challenges)
9. [Environment & Setup](#9-environment--setup)
10. [Data Flow & Data Contracts](#10-data-flow--data-contracts)
11. [Known Problems & Technical Debt](#11-known-problems--technical-debt)
12. [Script-by-Script Reference](#12-script-by-script-reference)
13. [Research Continuation Opportunities](#13-research-continuation-opportunities)
14. [Operational Notes](#14-operational-notes)
15. [Final Internal Summary](#15-final-internal-summary)

---

## 1. Project Overview

### What This Thesis Is Attempting

This project investigates whether **3D Gaussian Splatting (3DGS)** can replace or substantially outperform classical volumetric CT reconstruction algorithms in the **sparse-view regime** — specifically, reconstructing a human brain CT volume from 75 cone-beam X-ray projections rather than the clinical standard of 700–2000.

The thesis title references orthographic rendering because earlier phases (1 and 2, see §3) explored *orthographic Gaussian projection* of a CT brain onto 2D planes — a distinct visualization angle from the ultimate reconstruction goal. By Phase 3 (the defence project), the focus shifted to **full 3D reconstruction** using the R²-Gaussian method (NeurIPS 2024, Ruyi Zha et al.), applied to a custom LADAF brain CT dataset.

### Core Research Question

Given N≤75 X-ray cone-beam projections of a human brain at evenly-spaced angles, can a differentiable 3D Gaussian scene representation trained end-to-end via X-ray rendering outperform classical CT reconstruction algorithms (FDK, SART, CGLS, ASD-POCS) in terms of 3D structural fidelity (PSNR, SSIM)?

### Why Gaussian Splatting for Medical Imaging

Classical CT reconstruction is a linear inversion problem. FDK (filtered backprojection) is closed-form but produces severe streak artifacts at low view counts. Iterative methods (SART, CGLS, ASD-POCS) improve quality but take hours and are non-differentiable. NeRF-based methods (SAX-NeRF, NAF) are differentiable but slow to converge.

3DGS offers an explicit, fully differentiable scene representation optimized directly via gradient descent:
- No neural network inference at render time (pure geometry)
- Fast convergence: 5–15 minutes on RTX 3090 (vs hours for NeRF)
- Adaptive density control: Gaussians automatically concentrate in high-density anatomy regions
- Physical interpretation: each Gaussian is a localized attenuation blob

### Core Novelty and Adaptation Challenges

Standard 3DGS renders RGB images via alpha-compositing from perspective cameras. CT adaptation requires:

1. **New rendering equation**: X-ray follows Beer-Lambert line integral (additive attenuation), not alpha-compositing. The tile rasterizer's accumulation model must be changed from multiplicative opacity to additive density.
2. **No spherical harmonics**: CT density is view-independent. The entire SH color pipeline in the CUDA rasterizer is dead-code disabled.
3. **Scanner-derived camera geometry**: No COLMAP. Cameras come from physical scanner parameters (DSD, DSO, detector size), not photographic calibration.
4. **Second CUDA kernel needed**: Standard 3DGS only renders 2D projections. CT also needs 3D volumetric extraction (voxelization) for evaluation — a custom `GaussianVoxelizer` kernel was added.
5. **Coordinate system reconciliation**: Three distinct coordinate systems (TIGRE, PyTorch/NumPy, CUDA GLM) must all agree.
6. **Numerical stability via scene normalization**: Physical scanner distances (meters) are normalized to `[-1,1]³` before training.

---

## 2. Repository Structure

```
C:\Personal_Endeavours\Thesis\
│
├── HANDOFF.md                          ← Top-level archive handoff (portable audit record)
├── INTERNAL_HANDOFF.md                 ← This document
│
├── Misc\                               ← Loose assets
│   ├── evaluation_metrics.csv          ← Training metrics log (PSNR/SSIM over iterations)
│   ├── training.png                    ← Training curve screenshot
│   ├── GaSpCT_*.pdf                    ← Reference paper: Gaussian Splatting for CT
│   ├── Gaussian Splatting for Cinematic Anatomy.pdf
│   ├── Multi-Layer Gaussian Splatting for Immersive Anatomy Visualization.pdf
│   ├── SWE_Orthographic Gaussian Splatting_Presentation.pptx/pdf
│   ├── SWE_Orthographic Gaussian Splatting_Report.pdf
│   ├── Title_of_the_Thesis.pdf         ← Formal thesis title page
│   └── 0001-0546.mkv, 0001-0811.mkv   ← Video recordings (screen captures)
│
├── Thesis_PreDefence\                  ← Phase 1+2: cinematic-gaussians exploration
│   ├── .git\                           ← Wrapper repo (commits: 8c9b1a5, b5a0cbc)
│   ├── Brain1_JP2\                     ← LADAF 18.048µm brain, 1806 JP2 slices [LARGE]
│   ├── Brain2_JP2\                     ← LADAF 202.0µm brain, 811 JP2 slices [LARGE]
│   ├── JP2 to PNG converter\
│   │   ├── full_converter.py           ← Batch JP2→PNG pipeline (1806 slices)
│   │   ├── tif_to_png.py               ← TIF→PNG supplementary
│   │   └── venv\                       ← Portable venv (activate.bat patched)
│   ├── Napari Viewer(Not a must have)\
│   │   ├── view_3d.py                  ← Napari 3D brain slice viewer
│   │   ├── requirements.txt
│   │   └── brainview\                  ← Portable venv
│   └── cinematic-gaussians\            ← Phase 1+2 fork (own git: commit c7baa94)
│       ├── train.py                    ← Modified training (Windows multiprocessing fix)
│       ├── dataset_readers.py          ← Fixed: explicit xyz/rgb/opacity fields
│       ├── export_ply.py               ← Custom: export trained model to PLY
│       ├── make_cameras.py             ← Custom: generate orthographic camera JSON
│       └── Training_Data_v*/           ← COLMAP-prepared training runs [gitignored]
│
└── Thesis_Defence\                     ← Phase 3: R²-Gaussian CT reconstruction [MAIN]
    ├── .gitignore                      ← Excludes data/, output/, Brain2_PNG/, etc.
    ├── gaussian-splatting\             ← Upstream 3DGS baseline (reference only)
    ├── simple-knn\                     ← Top-level standalone simple-knn (reference copy)
    └── r2_gaussian\                    ← MAIN CODEBASE ← start here
        ├── train.py                    ← PRIMARY ENTRY POINT
        ├── test.py                     ← Evaluation
        ├── initialize_pcd.py           ← FDK-bootstrapped init
        ├── png_stack_to_npy.py         ← Preprocessing (custom-written)
        ├── r2_gaussian/                ← Core library package
        │   ├── arguments/              ← ModelParams, OptimizationParams, PipelineParams
        │   ├── dataset/                ← Scene, Camera, dataset_readers
        │   ├── gaussian/               ← GaussianModel, render(), query()
        │   ├── utils/                  ← ct_utils, loss_utils, image_utils, plot_utils...
        │   └── submodules/
        │       ├── simple-knn/         ← CUDA KNN extension
        │       └── xray-gaussian-rasterization-voxelization/   ← CRITICAL CUDA KERNEL
        ├── data_generator/             ← TIGRE-based synthetic CT generation
        ├── scripts/                    ← Batch training, baselines, format conversion
        ├── 3D_vis/                     ← 30+ anatomy-specific visualization scripts
        ├── brain/                      ← Brain-specific notebooks/renders
        ├── output/                     ← Training outputs [gitignored, present on disk]
        │   ├── Brain2_training/        ← Primary 30k-iter result (COMPLETE)
        │   └── Brain2_training_v2/     ← 60k-iter attempt (ABANDONED at 5k)
        ├── dummy_*.py                  ← Debug/bringup stubs
        ├── pickle_to_ply*.py           ← Model format conversion (3 versions)
        ├── PLYcomparer.py              ← Gaussian cloud diff tool
        ├── ASCII_to_bin.py             ← PLY format converter
        └── environment.yml             ← Conda environment spec
```

### Folder Relationships and Workflow Order

```
Thesis_PreDefence (Phase 1+2) is SEPARATE from Thesis_Defence (Phase 3).
They share the brain data source but have entirely different codebases.

PHASE 3 WORKFLOW ORDER:
  Brain2_JP2/ (raw LADAF) 
    → full_converter.py (JP2→PNG)
    → Brain2_PNG/ (PNG stack, gitignored)
    → png_stack_to_npy.py
    → Brain2_PNG.npy (256³ float32 volume)
    → data_generator/synthetic_dataset/generate_data.py (TIGRE forward projection)
    → cone_ntrain_75_angle_360/Brain2_PNG_cone/ (projections + meta_data.json)
    → initialize_pcd.py (FDK → sampled init cloud)
    → init_Brain2_PNG_cone.npy (50000×4 init positions)
    → train.py (full training)
    → output/Brain2_training/ (trained Gaussian pickle)
    → test.py (evaluation → PSNR/SSIM, .nii.gz)
    → 3D_vis/ scripts (visualization)
```

---

## 3. Research Progression — Three Phases

### Phase 1 — cinematic-gaussians Initial Exploration
**Folder:** `Thesis_PreDefence/cinematic-gaussians/` (git commit `5392c34`)  
**What happened:** Cloned and ran KeKsBoTer/cinematic-gaussians — a TUM fork of standard 3DGS designed for anatomy visualization. Ran on Brain1 LADAF data (18.048µm, 1806 slices). No code changes beyond getting it to execute. COLMAP was used for camera calibration from the PNG slice stack. The earliest brain render artifact is preserved as `1e523224-d008-4a29-b283-2d0ad1354feb-Copy.png`.  
**What it demonstrated:** That standard 3DGS (with COLMAP) can produce *some* visual structure from CT slice stacks, but the perspective-camera assumption is wrong for CT (CT uses parallel or cone-beam geometry).

### Phase 2 — Pipeline Construction & Windows Fixes
**Folder:** `Thesis_PreDefence/cinematic-gaussians/` (git commit `c7baa94`)  
**What happened:** Substantial code work on top of cinematic-gaussians:
- `dataset_readers.py`: Fixed implicit field handling for xyz/rgb/opacity — the upstream assumed a particular PLY field ordering that broke with this dataset.
- `train.py`: Fixed Windows multiprocessing crash (Python `lambda` in DataLoader is not picklable on Windows; replaced with named `collate_identity()` function).
- Added `export_ply.py`: Export trained 3DGS model to PLY format for external viewing.
- Added `make_cameras.py`: Generate orthographic camera JSON from a PNG stack — this is the "orthographic Gaussian" angle referenced in the thesis title.
- Added `full_converter.py`: Batch JPEG-2000 to PNG conversion pipeline for the LADAF data.
- Multiple training runs (`Training_Data_v*/`, `Test_Model_v*/`) with increasingly refined COLMAP setups.  
**What it demonstrated:** Orthographic rendering from a PNG-slice-derived Gaussian model is feasible. However, the COLMAP camera calibration approach (treating CT slices as photographs) is a geometrically wrong approximation.

### Phase 3 — CT-Native Reconstruction (MAIN)
**Folder:** `Thesis_Defence/r2_gaussian/` (git commit `bc76c2d`)  
**What happened:** Switched entirely to R²-Gaussian (Ruyi-Zha/r2_gaussian), which natively handles CT geometry via TIGRE. No COLMAP. Cameras come from scanner physics. This is the architecturally correct approach.
- Integrated custom brain dataset (LADAF Brain2, 202µm resolution) into the R²-Gaussian pipeline
- Built `png_stack_to_npy.py` (custom preprocessing script not in original R²-Gaussian)
- Modified `initialize_pcd.py` and `data_generator/initialize_pcd_all.py` for brain-specific initialization
- Built 34-script `3D_vis/` visualization suite for thesis presentation
- Built PLY/pickle conversion and inspection tooling
- Ran and logged full training experiment (Brain2_training, 30k iterations)  
**Current state:** Experiment complete. 3D reconstruction metrics are poor (see §7). Cause is likely scanner parameter mismatch + disabled densification, not a code bug.

---

## 4. Architecture Overview

### Full System Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│ INPUT: LADAF Brain CT                                                │
│  Brain2_JP2/ (811 × 202µm JP2 slices, JPEG-2000)                   │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ full_converter.py (JP2→PNG)
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ PREPROCESSING                                                        │
│  Brain2_PNG/ (PNG stack, sorted by slice index)                     │
│     ↓ png_stack_to_npy.py                                           │
│  vol.npy — float32, 256³, normalized [0,1], isotropic resample     │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ SYNTHETIC PROJECTION GENERATION (TIGRE)                             │
│  data_generator/synthetic_dataset/generate_data.py                  │
│     Inputs: vol.npy + scanner YAML (DSD=7, DSO=5, 512×512 det.)    │
│     Forward projection: tigre.Ax() at 75 angles over 360°          │
│     Noise: Poisson(10000) + Gaussian(σ=10)                         │
│     Outputs:                                                         │
│       proj_train/*.npy  — 75 × (512×512) float32                   │
│       proj_test/*.npy   — 100 × (512×512) float32                  │
│       vol_gt.npy        — ground truth 256³ volume                  │
│       meta_data.json    — scanner config + file manifest            │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ INITIALIZATION                                                       │
│  initialize_pcd.py                                                   │
│     FDK reconstruction → threshold at 0.05 → sample 50k voxels     │
│     Density rescaled by 0.15 (empirical)                            │
│     Output: init_Brain2_PNG_cone.npy  shape (50000, 4)              │
│              columns: [x, y, z, density] in [-1,1]³ world space    │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ GAUSSIAN OPTIMIZATION                                                │
│  train.py                                                            │
│                                                                      │
│  GaussianModel: N×{xyz, density, scale(3), rotation(4)}             │
│                                                                      │
│  Per iteration (30,000 total):                                       │
│   1. Pop random training camera (angle → c2w via angle2pose())      │
│   2. GaussianRasterizer CUDA → rendered X-ray (512×512, 1-channel)  │
│   3. L1 + λ·(1−SSIM) + λ·TV3D loss                                 │
│   4. loss.backward() → Adam step                                    │
│   5. Adaptive control: clone/split/prune [DISABLED in Brain2]       │
│                                                                      │
│  Checkpoints: .pth at specified iters                               │
│  Model saves: .pickle at specified iters                            │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ EVALUATION                                                           │
│  test.py                                                             │
│   2D: Render all train+test cameras → PSNR, SSIM per projection     │
│   3D: GaussianVoxelizer CUDA → 256³ density grid                   │
│       → compare to vol_gt.npy → PSNR3D, SSIM3D (per-axis)         │
│   Exports: eval*.yml, vol_pred.npy, vol_pred.nii.gz (3D Slicer)   │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ VISUALIZATION                                                        │
│  3D_vis/: marching cubes → matplotlib 3D mesh (anatomy-specific)    │
│  dummy_renderer.py: multi-angle PNG render from pickle              │
│  scripts/visualize_scene.py: Open3D camera + volume mesh viewer     │
│  plot_utils.py: show_two_volume() slider-based slice comparison     │
└──────────────────────────────────────────────────────────────────────┘
```

### Rendering Engine — Two CUDA Kernels

The CUDA extension `xray-gaussian-rasterization-voxelization` (submodule of `r2_gaussian/r2_gaussian/submodules/`) implements two distinct forward+backward passes:

| Kernel | Location | Purpose | Thread Layout |
|--------|----------|---------|---------------|
| `GaussianRasterizer` | `cuda_rasterizer/` | 2D X-ray projection (training) | 16×16 pixel tiles |
| `GaussianVoxelizer` | `cuda_voxelizer/` | 3D volume extraction (eval + TV loss) | 8×8×8 voxel blocks |

Both are fully differentiable (custom CUDA backward passes).

---

## 5. Technical Components — Deep Dive

### 5.1 GaussianModel (`r2_gaussian/gaussian/gaussian_model.py`)

**Purpose:** The learnable scene representation. Stores and optimizes the Gaussian cloud.

**Inputs:** `scale_bound` (tuple, world units)  
**Outputs:** Properties accessed via `get_xyz`, `get_density`, `get_scaling`, `get_rotation`, `get_covariance`

**4 trainable parameters per Gaussian:**

| Attr | Raw storage | Activation | Range |
|------|-------------|------------|-------|
| `_xyz` | raw floats | identity | free (bounded by bbox prune) |
| `_density` | pre-softplus | `Softplus` | [0, +∞) |
| `_scaling` | pre-sigmoid | `sigmoid * (smax−smin) + smin` | [scale_min, scale_max] |
| `_rotation` | unit quaternion | `L2 normalize` | unit sphere |

**Critical — Brain2 scale bounds:**
```python
scale_bound = [0.0005, 0.5] * 2.0  # = [0.001, 1.0] world units
```
This prevents Gaussians from collapsing to points or expanding to fill the entire volume.

**Adaptive density control (`densify_and_prune`):**
- Clone: small Gaussians (scale ≤ `densify_scale_threshold`) with high 2D gradient norm → duplicate + halved density
- Split: large Gaussians (scale > threshold) with high gradient → split into 2 with scale÷0.8
- Prune: Gaussians with density < `density_min_threshold` OR outside bbox OR too large on screen

**IMPORTANT:** In Brain2 experiments, densification is **disabled** by setting:
```yaml
densify_from_iter: 35000
densify_until_iter: 35000
```
Both are beyond the 30k training budget. Gaussian count stays fixed at 50000.

**Model persistence format:**
- `.pickle` — raw pre-activation parameters (`_xyz`, `_density`, `_scaling`, `_rotation`, `scale_bound`)
- NOT a standard PLY — cannot be opened by mesh viewers directly
- See `pickle_to_ply_v3.py` for conversion (but note the normalization bug — see §11)

**Dependencies:** `simple_knn._C` (distCUDA2 for initial distance estimation)

---

### 5.2 X-ray Rasterizer (`cuda_rasterizer/`)

**Purpose:** Forward-project all Gaussians to a 2D X-ray image at a given camera pose.

**Forward pass per Gaussian:**
1. Transform center to view space: `t = viewmatrix @ xyz`
2. Compute EWA 2D covariance projection (Zwicker 2002, Eq. 29/31):
   - `cov2D = J @ (T_view @ cov3D @ T_view.T) @ J.T`
   - Parallel beam: `J ≈ diag(focal_x, focal_y, 1)` (identity-like)
   - Cone beam: full perspective Jacobian
3. Invert cov2D → conic params (a, b, c)
4. Compute screen-space radius; bin into 16×16 pixel tiles
5. Sort all Gaussians by (tile_id | depth)

**Rendering loop (additive accumulation):**
```
rendered[pixel] = Σ density_i · exp(-0.5 · r^T · conic_i · r)
```
where r = pixel_center − projected_mean_i. **This is additive, not alpha-composited.**

**Single channel (`NUM_CHANNELS=1`):** Rendered image is a scalar per pixel.

**SH code:** Still present in `forward.cu` (`computeColorFromSH`) but commented as `#! We dont need this.` — dead code, never called.

---

### 5.3 GaussianVoxelizer (`cuda_voxelizer/`)

**Purpose:** Accumulate Gaussian contributions into a 3D voxel grid. Used for:
1. 3D PSNR/SSIM evaluation (compare to `vol_gt.npy`)
2. 3D TV loss computation during training (random 32³ sub-volume)

**Thread layout:** 8×8×8 voxels per CUDA block. For each voxel, accumulate all overlapping Gaussians:
```
vol[x,y,z] = Σ density_i · exp(-0.5 · dist3D^T · cov3D_inv · dist3D)
```

**Differentiable:** Full CUDA backward pass propagates gradients back to `_density`, `_scaling`, `_rotation`.

**Parameters:** `nVoxel_x/y/z`, `sVoxel_x/y/z`, `center_x/y/z` — must match scanner config.

---

### 5.4 Scene Loading (`r2_gaussian/dataset/`)

**`dataset_readers.py:readBlenderInfo()`** — Entry point for meta_data.json format.

**Scene normalization (applied once at load time):**
```python
scene_scale = 2 / max(meta_data["scanner"]["sVoxel"])  # = 1.0 for Brain2
```
Scales all spatial quantities: `sVoxel`, `dVoxel`, `sDetector`, `dDetector`, `DSD`, `DSO`, `offOrigin`, `offDetector`.
Projection intensities are also multiplied by `scene_scale`.

**Camera pose from angle (`angle2pose()`):**
```python
R = R_z(θ) @ R_z(+90°) @ R_x(-90°)
T = [DSO·cos(θ), DSO·sin(θ), 0]
```
This produces a c2w matrix where the X-ray source orbits the origin at radius DSO.

**`dDetector` convention (CRITICAL):**
- `dDetector = [v_size, u_size]` — row-first (v), then column (u)
- `FovX` computed from `dDetector[1]` (u), `FovY` from `dDetector[0]` (v)
- **Do not flip this.** It will break every camera's projection.

---

### 5.5 Loss Functions (`r2_gaussian/utils/loss_utils.py`)

Three loss terms:

| Term | Weight in Brain2 | Domain | Purpose |
|------|------------------|--------|---------|
| L1 | 1.0 | 2D projection | Data fidelity |
| 1−SSIM | λ=0.2 | 2D projection | Multi-scale structural match |
| TV 3D | λ=0.05 | 3D (32³ sub-vol) | Piecewise-smooth CT prior |

TV loss samples a random 32³ sub-region each iteration to avoid querying the full 256³ grid:
```python
tv_vol_center = (bbox[0] + tv_vol_sVoxel/2) + (bbox[1] - tv_vol_sVoxel - bbox[0]) * torch.rand(3)
```

---

### 5.6 Optimization (`train.py`)

**Optimizer:** Adam, `eps=1e-15`, `zero_grad(set_to_none=True)`

**Per-group exponential LR decay:**
```
lr(t) = exp(log(lr_init)·(1−t) + log(lr_final)·t)  where t = step/max_steps
```

| Parameter | LR init | LR final | Note |
|-----------|---------|---------|------|
| xyz | 0.00016 | 1.6e-6 | Brain2 custom (not default 0.0002) |
| density | 0.01 | 0.001 | |
| scaling | 0.005 | 0.0005 | |
| rotation | 0.001 | 0.0001 | |

**Camera sampling:** Pop from shuffled stack; replenish when empty. `shuffle=False` for Brain2 — cameras are consumed in order.

**Seeds:** `random.seed(0)`, `np.random.seed(0)`, `torch.manual_seed(0)` — results are reproducible if environment matches.

---

### 5.7 Data Generator (`data_generator/synthetic_dataset/`)

**`generate_data.py`** — Core synthetic data creation using TIGRE:
```python
projs_train = tigre.Ax(
    np.transpose(vol, (2,1,0)).copy(),  # TIGRE wants (Z,Y,X) → transpose
    geo, angles
)[:, ::-1, :]  # flip vertical axis for detector convention
```
Note: The `(2,1,0)` transpose and `[::-1]` flip are critical. These reconcile NumPy (D,H,W) with TIGRE's internal (Z,Y,X) convention and the detector y-axis orientation.

**Noise model:**
- Poisson: simulates photon counting noise with I₀=10000 (moderate-noise regime)
- Gaussian: σ=10 (additive electronic noise)
- Note: field is named `possion_noise` (typo) in `meta_data.json` — code reads it correctly

---

### 5.8 TIGRE Bridge (`r2_gaussian/utils/ct_utils.py`)

**`get_geometry_tigre(cfg)`** — Converts our scanner config dict into TIGRE `Geometry` object.

**Critical ordering differences:**
- Our config stores dimensions as `[x, y, z]` or `[u, v]`
- TIGRE expects `nVoxel` in `(Z,Y,X)` order → reversed: `cfg["nVoxel"][::-1]`
- `offDetector` reordering: `[v, u, 0]` not `[u, v, 0]`

If you modify scanner geometry, verify TIGRE axis ordering first. A transpose in the wrong direction silently produces a valid but wrong reconstruction.

---

## 6. Experimental Pipeline

### Dataset — LADAF Brain2

| Property | Value |
|----------|-------|
| Source | LADAF-2021-17 (Laboratoire d'Anatomie des Alpes Françaises) |
| Resolution | 202.0 µm isotropic |
| Raw format | 811 × JPEG-2000 slices |
| Working format | 256³ float32 `.npy` (normalized 0–1, percentile-clipped) |
| Projection count (train) | 75 (cone-beam, 360°, evenly-spaced) |
| Projection count (test) | 100 (random angles within 360°) |
| Projection resolution | 512×512 per projection |
| Scanner geometry | DSD=7m, DSO=5m, sDetector=4×4m, sVoxel=2×2×2m |

Note: Physical dimensions in meters are large because they represent scanner geometry scaled from actual mm values. After scene normalization (`scene_scale = 1.0` since `max(sVoxel)=2.0`), everything maps cleanly to `[-1,1]³`.

### Preprocessing Pipeline

```
Step 1: JP2 → PNG
  Tool: full_converter.py (Pillow, glymur)
  Input: Brain2_JP2/ (811 JPEG-2000 files)
  Output: Brain2_PNG/ (811 PNG files, sorted numerically)

Step 2: PNG → .npy
  Tool: png_stack_to_npy.py
  Key operations:
    - Luminosity grayscale: 0.2989R + 0.5870G + 0.1140B
    - Percentile normalization: clip to [1st, 99th] percentile → [0,1]
    - Isotropic resample to 256³ (skimage resize, order=1, anti-aliasing)
  Output: Brain2_PNG.npy (256, 256, 256) float32

Step 3: .npy → projection dataset
  Tool: data_generator/synthetic_dataset/generate_data.py
  Key: TIGRE tigre.Ax() forward projection
  Output: cone_ntrain_75_angle_360/Brain2_PNG_cone/ (complete dataset dir)

Step 4: Initialization
  Tool: initialize_pcd.py --recon_method fdk --n_points 50000
  Key: FDK via TIGRE, threshold=0.05, density_rescale=0.15
  Output: init_Brain2_PNG_cone.npy (50000, 4)
```

### Hyperparameters (Brain2 experiment)

```yaml
# From output/Brain2_training/cfg_args.yml
iterations: 30000
lambda_tv: 0.05
lambda_dssim: 0.2
position_lr_init: 0.00016        # Custom (default: 0.0002)
position_lr_final: 1.6e-06
position_lr_max_steps: 30000
density_lr: 0.01
scaling_lr: 0.005
rotation_lr: 0.001
scale_min: 0.0005
scale_max: 0.5
max_num_gaussians: 200000
densify_from_iter: 35000          # Beyond training budget — effectively disabled
densify_until_iter: 35000         # Same
densify_grad_threshold: 5e-05
density_min_threshold: 0.005
tv_vol_size: 32
debug: true                       # SLOWDOWN — should be false
```

### Training Flow

1. Load `meta_data.json` → parse scanner config → normalize scene to `[-1,1]³`
2. Load 75 training cameras (Camera objects with c2w, FovX/Y, projection matrix)
3. Load `init_Brain2_PNG_cone.npy` → initialize GaussianModel (50000 Gaussians)
4. Setup Adam optimizer with 4 parameter groups + LR schedulers
5. Training loop (30000 iterations, ~46 minutes total on observed timeline)
6. At test iterations [1, 5000, 10000, 20000, 30000]: evaluate 2D + 3D metrics
7. Save `.pickle` and `.pth` checkpoints at `save_iterations`

### Evaluation Methodology

**2D evaluation (`metric_proj`):**
- Render all train cameras and test cameras separately
- PSNR: per-projection, averaged
- SSIM: per-projection (max-normalized per slice), averaged
- Note: per-slice max normalization inflates SSIM slightly (removes absolute scale)

**3D evaluation (`metric_vol`):**
- Call `query()` → extract full 256³ volume from Gaussians
- PSNR: pixel-wise against vol_gt.npy
- SSIM: computed along each of 3 axes, averaged per axis

**Logged metrics CSV** (`Misc/evaluation_metrics.csv`):

| Milestone | PSNR 3D | SSIM 3D | PSNR 2D | SSIM 2D |
|-----------|---------|---------|---------|---------|
| Iter 1 | 5.081 | 0.308 | 8.553 | 0.345 |
| Iter 1500 | 6.222 | 0.014 | 9.171 | 0.350 |
| Iter 30000 | 6.222 | 0.014 | 11.005 | 0.557 |

**Observation:** SSIM3D drops from 0.308 at iter 1 to 0.014 at iter 1500 and stays there. This is pathological — the initialization is already better structured than the trained model in some sense. The 2D PSNR/SSIM improve monotonically, showing the model is fitting projections but not recovering the 3D structure.

---

## 7. Research Findings

### What Worked

- **Projection fitting (2D):** The model successfully minimizes 2D projection loss. SSIM2D reaches 0.557 at 30k iterations, showing the Gaussian cloud can produce projections that are structurally similar to CT projections of brain data.
- **Training stability:** Loss decreases monotonically. No NaN/divergence. The numerical normalization scheme (scene_scale, density_rescale) is effective.
- **Visualization pipeline:** The `3D_vis/` suite and `dummy_renderer.py` produce compelling visualizations of the Gaussian cloud structure that were used in thesis presentations.
- **Format conversion pipeline:** `pickle_to_ply_v3.py`, `ASCII_to_bin.py`, PLYcomparer — these work (with the normalization bug caveat) and allow external inspection.
- **TIGRE integration:** `dummy_TIGRE_run.py` confirms TIGRE is functional. Forward projection and FDK reconstruction both work on the Brain2 data.
- **Phase 2 cinematic-gaussians:** The orthographic camera generation (`make_cameras.py`) and Windows multiprocessing fix (`collate_identity`) both work and produced successful training runs.

### What Failed

- **3D reconstruction quality is poor:** `psnr_3d = 6.22 dB` is near-noise-floor. `ssim_3d = 0.014` indicates near-zero structural agreement between reconstructed and ground-truth volume. This is the central problem of the project.
- **SSIM3D collapse from iter 1 to 1500:** A 22× drop in SSIM3D (0.308 → 0.014) in the first 1500 iterations suggests the initial FDK-sampled Gaussian cloud has reasonable spatial structure that the early training iterations immediately destroy, possibly due to the TV loss or density rescaling pushing Gaussians away from the correct positions.
- **v2 experiment abandoned:** The 60k-iteration run with densification re-enabled was abandoned after 5k iterations showed no improvement over the v1 30k final result. No root-cause analysis was recorded.
- **Densification disabled:** Adaptive density control (the mechanism by which 3DGS achieves high-quality reconstructions in standard 3DGS tasks) was disabled. This likely limits the model's expressiveness — 50k fixed Gaussians may be insufficient to capture brain soft tissue detail.

### Bottlenecks

1. **Scanner parameter mismatch (highest likelihood root cause):** The scanner YAML uses DSD=7m, DSO=5m, sVoxel=2m. These are the R²-Gaussian paper's default values for their benchmark datasets, not calibrated to Brain2. Brain CT typically has much smaller physical dimensions. If sVoxel is wrong relative to the actual brain size, the scene normalization maps the volume to the wrong scale, and all Gaussian positions are fundamentally off.
2. **FDK initialization on brain soft tissue:** FDK is well-suited for bony structures (high contrast). Brain soft tissue has very low X-ray contrast and noisy projections. The FDK reconstruction of brain at 75 views with Poisson noise may produce a near-uniform cloud that gives the Gaussians no meaningful starting point.
3. **`debug: true` in training config:** Enables extra CUDA assertions, causing ~2–5× slowdown. All Brain2 experiments were run in debug mode.

### Unexpected Findings

- The metric CSV shows PSNR3D is already at 6.22 by iter 1500 and **does not change at all** through iter 30000. 2D metrics keep improving throughout. This means the model is fitting the projections in a way that is entirely detached from the 3D structure — the projection-space loss provides no gradient signal that recovers 3D density. This is a fundamental ill-posedness problem when using only 2D loss on a 3D representation.
- The TV loss (lambda=0.05) may be actively harmful: it pushes for piecewise-constant density, but at this resolution with incorrect initialization, it may be suppressing the Gaussian cloud from exploring the correct density manifold.

### Quality and Computational Limitations

- Single GPU, single scene — no multi-GPU or batch-scene support
- 256³ volume at float32 is ~64MB — fits comfortably in VRAM
- Training time: ~46 minutes for 30k iterations (from CSV timestamp: 19:37 → 20:23)
- The visualization scripts (especially `show_gaussians()`) are O(N) with Open3D mesh creation and become unusably slow for N>10k Gaussians

---

## 8. Engineering Challenges

### Memory Pressure
- 50k Gaussians × 4 params × float32 is manageable (~3MB)
- The GaussianVoxelizer querying 256³ volume at every test iteration requires a full 3D grid allocation: 256³ × float32 = 64MB per query
- TV loss with 32³ sub-volume per iteration: negligible
- If densification is re-enabled and Gaussians grow to 200k, VRAM pressure increases significantly

### Sparse-View Reconstruction (Central Problem)
- 75 projections over 360° is a severely underdetermined inverse problem
- The Gaussian parameterization has 50000 × 7 = 350000 free parameters
- The data has 75 × 512 × 512 = ~19.7M measurements, but only ~75 independent projection angles
- The effective degrees of freedom per angle is much lower than the projection resolution — this is the ill-posedness source

### Volumetric Inconsistency
- 2D projection loss does not guarantee 3D consistency
- The Radon transform has a large null space: many 3D volumes project identically at 75 angles
- Without a strong 3D prior or sufficient views, the optimizer can find a projection-consistent but volumetrically wrong solution

### Gaussian Instability (Potential, Not Observed Here)
- When densification is enabled with wrong thresholds, Gaussian count can explode or collapse
- The `density_min_threshold` prune can remove valid Gaussians if density activations are not properly initialized
- In Brain2, this was avoided by disabling densification entirely — but this is a workaround, not a solution

### CT Normalization Pipeline
- Three different normalization steps interact: percentile normalization in `png_stack_to_npy.py`, scene_scale in `dataset_readers.py`, and density_rescale in `initialize_pcd.py`
- If any of these three are applied inconsistently when processing new data, the volume and Gaussian scales will be misaligned
- The scanner YAML physical dimensions must be calibrated to match the actual acquisition — using default values from benchmark datasets is a common error source

### Rendering Artifacts
- The EWA splatting approximation for X-ray is theoretically inexact for Gaussians with large depth extent relative to their 3D covariance — this causes "blurry" contributions to projections
- The additive accumulation model does not account for self-shadowing (Beer-Lambert exponential) — for optically thin materials (air, soft tissue) this is acceptable; for bone it may introduce systematic errors

### Convergence Issues
- The position LR was manually tuned to 80% of default (0.00016 vs 0.0002) without documented justification — this may be suboptimal
- No LR warmup — training starts at the initial LR immediately

### Preprocessing Pain Points
- TIGRE expects `(Z,Y,X)` array ordering but NumPy volumes are typically `(D,H,W)` — the `np.transpose(vol, (2,1,0))` call in `generate_data.py` is non-obvious and silently wrong if omitted
- The detector flip `[:, ::-1, :]` in `generate_data.py` and `ct_utils.py` is a detector orientation convention — without it, projections are vertically mirrored

---

## 9. Environment & Setup

### Conda Environments (3 total)

| Env | Project | Python | PyTorch | CUDA | Lock file |
|-----|---------|--------|---------|------|-----------|
| `cin3dgs` | Thesis_PreDefence/cinematic-gaussians | 3.12 | 2.5.1 | 12.1 | `cinematic-gaussians/environment.yml` |
| `r2_gaussian` | Thesis_Defence/r2_gaussian | 3.9 | 1.12.1+cu118 | 11.8 | `r2_gaussian/environment.yml` (→ `r2_gaussian_new`) |
| `gaussian_splatting` | Thesis_Defence/gaussian-splatting | — | — | — | `gaussian-splatting/environment.yml` |

### Phase 3 (r2_gaussian) Setup — Step by Step

```bash
# 1. Create conda environment
conda env create --file r2_gaussian/environment.yml
conda activate r2_gaussian_new

# 2. Install PyTorch (NOT in yml — must add separately)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 3. Windows only
SET DISTUTILS_USE_SDK=1

# 4. Build X-ray rasterizer/voxelizer CUDA extension
pip install -e r2_gaussian/r2_gaussian/submodules/xray-gaussian-rasterization-voxelization \
    --no-build-isolation

# 5. Build simple-knn CUDA extension
pip install -e r2_gaussian/r2_gaussian/submodules/simple-knn \
    --no-build-isolation

# 6. Install TIGRE (GPU CT reconstruction)
# Download from https://github.com/CERN/TIGRE/archive/refs/tags/v2.3.zip
pip install TIGRE-2.3/Python --no-build-isolation

# 7. Verify TIGRE
python r2_gaussian/dummy_TIGRE_run.py
```

### Key Dependency Pins

- `numpy==1.24.1` — pinned; later versions break CUDA extension builds
- `open3d==0.18.0` — for visualization (not 0.19.0 which changes API)
- CUDA 11.8 must match system CUDA toolkit (check with `nvcc --version`)
- PyTorch and CUDA toolkit versions must match exactly — mismatches cause silent wrong results

### Python venvs (Phase 1+2 tools)

```bat
:: JP2 converter
cd "Thesis_PreDefence\JP2 to PNG converter"
venv\Scripts\activate.bat
python full_converter.py

:: Napari viewer
cd "Thesis_PreDefence\Napari Viewer(Not a must have)"
brainview\Scripts\activate.bat
python view_3d.py
```
Both `activate.bat` files use `%~dp0`-relative paths — portable, no hardcoded paths.

### GPU Requirements
- Minimum 16GB VRAM recommended (256³ volume + 50k Gaussians)
- `cuda:0` is hardcoded in `general_utils.py:safe_state()` — must be the training GPU
- Development machine: Windows 11, verified on (inferred) RTX 3090-class hardware

---

## 10. Data Flow & Data Contracts

### Expected Formats

**`meta_data.json` (required):**
```json
{
  "vol": "vol_gt.npy",
  "scanner": {
    "mode": "cone",          // or "parallel"
    "DSD": 7.0,              // Source-Detector distance [must match sVoxel scale]
    "DSO": 5.0,              // Source-Origin distance
    "nDetector": [512, 512], // [v, u] — row first
    "sDetector": [4.0, 4.0], // [v, u] physical size
    "nVoxel": [256, 256, 256],
    "sVoxel": [2.0, 2.0, 2.0],
    "dVoxel": [computed],    // sVoxel/nVoxel — auto-computed if missing
    "dDetector": [computed], // sDetector/nDetector — auto-computed if missing
    "offOrigin": [0,0,0],    // volume center offset
    "offDetector": [0,0],
    "totalAngle": 360.0,
    "startAngle": 0.0,
    "accuracy": 0.5,         // TIGRE forward projection accuracy
    "filter": "ram-lak"      // FDK filter
  },
  "proj_train": [
    {"angle": 0.0, "image_path": "proj_train/proj_train_0000.npy"},
    ...
  ],
  "proj_test": [...]
}
```

**`vol_gt.npy`:** shape `(256, 256, 256)`, dtype `float32`, range `[0, 1]`

**`init_<name>.npy`:** shape `(N, 4)`, dtype `float64`, columns `[x, y, z, density]` in `[-1,1]³` world space

**`proj_train/proj_train_XXXX.npy`:** shape `(512, 512)`, dtype `float32`, non-negative

**Gaussian pickle (`.pickle`):**
Python dict with keys: `_xyz`, `_density`, `_scaling`, `_rotation`, `scale_bound`, plus optimizer state

### Tensor Conventions

| Convention | Value |
|-----------|-------|
| Coordinate system | Right-handed, Y-up (TIGRE convention) |
| Scene normalization | `[-1,1]³`, applied once at `readBlenderInfo()` |
| Rotation storage | Unit quaternion `(w, x, y, z)` |
| R in world-to-view | **Transposed** relative to standard convention (GLM column-major inheritance) |
| Projection matrix | `getProjectionMatrix()` — parallel: identity; cone: perspective with znear=0.01 |
| Image layout | `(1, H, W)` tensors for single-channel X-ray |

### Normalization Assumptions

- `scene_scale = 2/max(sVoxel)` applied to ALL spatial quantities and projection intensities simultaneously
- For Brain2: `max(sVoxel) = 2.0` → `scene_scale = 1.0` (no-op in this specific case)
- `density_rescale = 0.15` applied once during FDK initialization — NOT applied during training
- Percentile normalization (1st–99th) applied during `png_stack_to_npy.py` — NOT reversible

### Coordinate System Diagram

```
World Space [-1,1]³:
  Origin = center of reconstructed volume
  X-ray source orbits the origin at radius DSO (after scaling)
  Camera looks at origin from source position

TIGRE Space:
  nVoxel ordered as (Z, Y, X) — reversed from world (X, Y, Z)
  offOrigin and offDetector require careful axis reordering (see ct_utils.py)

CUDA GLM Space:
  Column-major matrices
  Rotation stored as transpose of standard R
  Full proj transform = proj_matrix @ world_view_transform (both pre-transposed)
```

---

## 11. Known Problems & Technical Debt

### Hardcoded Absolute Paths (Critical for New Machine)

| Script | Hardcoded path | Severity |
|--------|---------------|---------|
| `dummy_renderer.py:21` | `output/Brain2_training/.../point_cloud.pickle` | High |
| `dummy_point_renderer.py:7` | Same | High |
| `dummy_init_pcd.py:6-7` | `vol_gt.npy` and output paths | High |
| `pickle_to_ply_v3.py:15-17` | `output/Brain2_training/...` | High |
| `ASCII_to_bin.py:6-8` | Same | High |
| `PLYcomparer.py:101-102` | Brain2 output + nonexistent `gaussian-splatting/output/room/input.ply` | High |
| `3D_vis/3D_vis.py:28` | `output/Brain2_training/...` | High |
| All `3D_vis/3D_vis_*.py` | Various hardcoded anatomy-specific paths | Medium |
| `cfg_args.yml` in output | `source_path: C:\Users\Mushfiq\Desktop\...` | High |

**Fix for cfg_args path issue:** Override with `-s <actual_path>` flag when calling `test.py`.

### Broken Code

1. **`PLYcomparer.py:102`** — references `gaussian-splatting/output/room/input.ply` which does not exist. Script will crash if the second comparison path is used.
2. **`pickle_to_ply_v3.py` normalization bug (lines 76–77):**
   ```python
   # WRONG: _xyz is already in world coordinates [-1,1]³
   points_normalized = bbox_min + points * scale
   ```
   The raw `_xyz` values are already normalized world coordinates. This re-normalization maps them to a wrong range. Use the xyz values directly.
3. **`forward.cu`** — `computeColorFromSH` is dead code (~200 lines). The comment `#! We dont need this.` marks it but it's never removed — adds confusion and compile time.

### Duplicated Logic

- **3 versions of `pickle_to_ply.py`** (`v1`, `v2`, `v3`) — incremental debugging artifacts. Only v3 is semi-canonical (but has the bug above). Delete v1 and v2.
- **2 `simple-knn` installations:** One at `r2_gaussian/r2_gaussian/submodules/simple-knn/`, another standalone at `Thesis_Defence/simple-knn/`. The submodule version is the one actually used.
- **`dummy_init_pcd.py`** reimplements initialization from `initialize_pcd.py` with different (wrong) hardcoded parameters (`dVoxel=[2,2,2]` instead of reading from scanner config). This is a dangerous alternative that will silently produce wrong init clouds.
- **All `3D_vis/3D_vis_*.py`** are near-identical — should be a single `3D_vis_generic.py --subject <name> --threshold <t>`.

### Hidden Coupling

- `train.py` calls `torch.cuda.set_device(torch.device("cuda:0"))` — fails on multi-GPU machines if GPU 0 is already occupied or the target is GPU 1.
- `dataset_readers.py` implicitly assumes `dDetector[0]` = v-dimension. If a future dataset stores `dDetector` as `[u, v]`, all projections will be geometrically wrong without any error message.
- `cfg_args` saved by training uses Python's `eval()` to parse the Namespace string — this is an arbitrary code execution risk if cfg files are shared.

### Missing Validation

- No check that `init_*.npy` was created with the same `scene_scale` as the current training run
- No check that `vol_gt.npy` dimensions match `nVoxel` in `meta_data.json`
- No assertion that projection intensities are non-negative after noise addition (partially handled: `projs_train[projs_train < 0.0] = 0.0`)
- No per-epoch or per-iteration convergence check — training runs the full 30k regardless

### Scaling Issues

- No multi-GPU support
- No distributed training
- `plot_utils.py:show_gaussians()` creates Open3D mesh objects in a Python loop over all Gaussians — O(N), unusable above ~10k
- `test.py:multithread_write` uses `max_workers=None` (unbounded thread pool) — can exhaust file handles for large datasets

---

## 12. Script-by-Script Reference

### Core Pipeline Scripts

**`r2_gaussian/train.py`**
- Pipeline stage: 4 (training)
- Command: `conda activate r2_gaussian; cd r2_gaussian; python train.py -s <data_path> -m <output_path> [--config <yaml>]`
- Assumes: Run from `r2_gaussian/` directory. CUDA device 0. Init `.npy` file present in data_path.
- Produces: `output/<name>/point_cloud/iteration_*/point_cloud.pickle`, `cfg_args.yml`, TensorBoard events, `eval/iter_*/eval*.yml`
- Watch for: OOM if Gaussians densify beyond VRAM budget; progress bar shows `pts` count

**`r2_gaussian/test.py`**
- Pipeline stage: 5 (evaluation)
- Command: `python test.py -m output/Brain2_training [--iteration 30000]`
- Assumes: `cfg_args` in model_path; trained pickle at specified iteration; `vol_gt.npy` accessible
- Produces: `test/iter_*/vol_pred.npy`, `vol_pred.nii.gz`, `proj_test_*.png`, `eval*.yml`
- Hidden: Reads source_path from cfg_args (may have old absolute path — override with `-s`)

**`r2_gaussian/initialize_pcd.py`**
- Pipeline stage: 3 (initialization)
- Command: `python initialize_pcd.py --data <data_path> [--recon_method fdk] [--n_points 50000]`
- Assumes: TIGRE installed; `meta_data.json` and `proj_train/` in data_path; `valid_voxels ≥ n_points`
- Produces: `init_<name>.npy` in data_path
- Critical params: `density_thresh=0.05`, `density_rescale=0.15` (both empirical for Brain2)
- Debug mode: Add `--evaluate` to print `3D PSNR for initial Gaussians` — should be ≥15 dB for good init

**`r2_gaussian/png_stack_to_npy.py`**
- Pipeline stage: 2 (preprocessing)
- Command: `python png_stack_to_npy.py <png_folder> <out.npy> [256] [spacing_z spacing_y spacing_x]`
- Produces: float32 `.npy` at target_size³, normalized to [0,1]
- Note: Custom script not in upstream R²-Gaussian — written for this thesis

**`r2_gaussian/data_generator/synthetic_dataset/generate_data.py`**
- Pipeline stage: 2b (CT simulation)
- Command: `python data_generator/synthetic_dataset/generate_data.py --vol <vol.npy> --scanner <yaml> --output <dir> --n_train 75 --n_test 100`
- Critical: TIGRE must be installed. The `(2,1,0)` transpose and `[::-1]` flip are required — do not remove.
- Produces: Complete dataset directory with `meta_data.json`, `proj_train/`, `proj_test/`, `vol_gt.npy`

### Visualization Scripts

**`r2_gaussian/3D_vis/3D_vis.py`** (and all anatomy variants)
- Standalone — no CLI args. Hardcoded pickle path.
- Must edit path before use. Marching cubes → matplotlib 3D window.
- Use `3D_vis_head.py` for Brain2 data.

**`r2_gaussian/scripts/visualize_scene.py`**
- Command: `python scripts/visualize_scene.py -s <data_path> [--mc_thresh 0.1]`
- Opens Open3D window with camera frustums + marching cubes mesh from `vol_gt.npy`
- Requires `vol_gt.npy` in source path
- Critical use: Run this BEFORE training to verify camera frustums actually enclose the volume

**`r2_gaussian/dummy_renderer.py`**
- Debug/demo script. Hardcoded pickle path.
- Saves multi-angle PNGs to `brain/elevation_*/angle_*.png`
- Not intended for production use

### Conversion & Utility Scripts

**`r2_gaussian/pickle_to_ply_v3.py`** (canonical, but has bug)
- Converts trained Gaussian pickle → ASCII PLY
- Bug: applies unnecessary re-normalization to xyz. Use raw `_xyz` values directly.

**`r2_gaussian/scripts/run_traditional_methods.py`**
- Runs FDK, SART, CGLS, ASD-POCS, OS-ASD-POCS via TIGRE for baseline comparison
- Input: NAF `.pickle` format (use `ours_to_naf_format.py` to convert)
- Produces: Per-method PSNR/SSIM in `eval_3d.yml`
- **This has not been run on Brain2 data yet** — no baseline comparison exists

**`r2_gaussian/PLYcomparer.py`**
- Compares two PLY files' Gaussian distributions
- Has hardcoded path to nonexistent `gaussian-splatting/output/room/input.ply` — will crash if used for cross-project comparison
- Can be useful if both paths are fixed manually

**`r2_gaussian/dummy_TIGRE_run.py`**
- Sanity check for TIGRE installation
- Run this first on any new machine before attempting data generation or initialization

### Phase 2 Scripts (cinematic-gaussians)

**`JP2 to PNG converter/full_converter.py`**
- Convert all 1806 Brain1 (or 811 Brain2) JP2 slices to PNG
- Uses `glymur` for JPEG-2000 and `Pillow` for PNG output
- Requires `venv` activation: `JP2 to PNG converter\venv\Scripts\activate.bat`

**`cinematic-gaussians/make_cameras.py`**
- Generates orthographic camera JSON from PNG stack
- This is the "orthographic" in the thesis title — creates parallel-projection cameras for the Gaussian model

**`cinematic-gaussians/export_ply.py`**
- Exports trained cinematic-gaussians model to PLY for external viewing

---

## 13. Research Continuation Opportunities

### Immediate Technical Fixes (Should Be Done Before Any New Experiments)

1. **Set `debug: false`** in all training configs. Expected 2–5× training speedup.
2. **Calibrate scanner parameters** to Brain2 actual acquisition geometry. Run `visualize_scene.py` and verify camera frustums enclose the brain mesh. The DSD=7m, DSO=5m values may be incorrect for this dataset.
3. **Fix `pickle_to_ply_v3.py`** normalization bug — remove the `bbox_min + points * scale` transformation for xyz.
4. **Run `initialize_pcd.py --evaluate`** on the current data and report 3D PSNR for the initial Gaussians. If below 15 dB, the FDK init is failing and must be fixed before training will converge.

### Next Experiments (Research Value)

1. **Enable densification with paper defaults:**
   ```yaml
   densify_from_iter: 500
   densify_until_iter: 15000
   densify_grad_threshold: 5e-5
   ```
   The NeurIPS 2024 paper shows densification is critical for quality. Run a 30k experiment with densification enabled on Brain2.

2. **Sparse-view study (core thesis result):**
   Generate Brain2 datasets at 10, 25, 50, 75, 100 views. Compare R²-Gaussian vs FDK/SART at each view count. This is the primary quantitative contribution the thesis needs.

3. **Run traditional baselines:**
   Execute `scripts/run_traditional_methods.py` on Brain2 data (after converting to NAF format). Without baseline numbers, the R²-Gaussian results cannot be contextualized.

4. **Verify scanner parameter calibration:**
   The LADAF 202µm scan has known physical dimensions. Convert to mm: 811 slices × 202µm = 163.8mm brain height. Check if the scanner YAML `sVoxel` (post-normalization) corresponds to realistic brain dimensions.

### Publication-Ready Directions

1. **Brain CT-specific Gaussian splatting analysis:** The `3D_vis/` suite generates compelling visualizations. A paper on applying R²-Gaussian to neuroimaging CT is a natural output, provided the 3D quality issue is resolved.
2. **Ablation on loss design:** λ_tv, λ_dssim, TV sub-volume size — none systematically evaluated. A thorough ablation on Brain2 is publishable.
3. **Parallel vs cone beam CT comparison:** The rasterizer supports both geometries. A direct comparison on the same dataset would be a novel contribution.
4. **Real CT data application:** `data_generator/real_dataset/` infrastructure exists but was never used. Applying to DICOM-format clinical brain scans (post-preprocessing) would be a strong follow-up.
5. **Orthographic rendering analysis:** The Phase 2 work on `make_cameras.py` and orthographic Gaussian projection has not been fully analyzed. There may be an independent contribution in comparing orthographic vs cone-beam representations.

---

## 14. Operational Notes

### First Steps on a New Machine

```bash
# 1. Check CUDA toolkit version
nvcc --version

# 2. Install conda, create r2_gaussian_new environment
conda env create -f r2_gaussian/environment.yml

# 3. Install PyTorch matching CUDA version
# For CUDA 11.8:
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 4. Build CUDA extensions (critical)
pip install -e r2_gaussian/r2_gaussian/submodules/xray-gaussian-rasterization-voxelization --no-build-isolation
pip install -e r2_gaussian/r2_gaussian/submodules/simple-knn --no-build-isolation

# 5. Verify TIGRE
python r2_gaussian/dummy_TIGRE_run.py

# 6. Verify data pipeline
python r2_gaussian/initialize_pcd.py --data <data_path> --evaluate
# Should print: "3D PSNR for initial Gaussians: XX.X"
```

### Reproducibility Gotchas

- **Path in cfg_args:** `output/Brain2_training/cfg_args.yml` contains `source_path: C:\Users\Mushfiq\Desktop\Thesis Defence\...`. Override with `-s` flag in `test.py`.
- **numpy pin:** `numpy==1.24.1` is required. Newer numpy versions change ABI in ways that break pre-compiled CUDA extensions. Do not upgrade.
- **Seed**: `random.seed(0)`, `np.random.seed(0)`, `torch.manual_seed(0)` in `general_utils.py:safe_state`. Results are deterministic if CUDA version and GPU architecture match.
- **`shuffle=False`** in `Scene.__init__()` for Brain2 — cameras processed in order. Default is `shuffle=True` for other datasets.
- **`debug: true`** in Brain2 config — disable for any timing or production run.

### GPU Requirements

- Minimum: RTX 3090 (24GB) for 256³ volume + 50k Gaussians
- `cuda:0` hardcoded — must be the target GPU
- Approximate memory budget: 8GB for 50k Gaussians, scales to ~32GB for 200k Gaussians + voxelizer

### Runtime Expectations

- PNG → npy preprocessing: ~2 minutes for 811 slices
- TIGRE data generation (75 projections): ~5 minutes
- FDK initialization (50k points): ~3 minutes
- Training (30k iterations, 50k Gaussians, debug=false): ~20 minutes
- Training (30k iterations, 50k Gaussians, debug=true): ~46 minutes (observed)
- Evaluation pass (render + voxelize): ~2 minutes

### Debugging Advice

1. **Volume misalignment**: Use `show_two_volume()` from `plot_utils.py` to compare `vol_gt.npy` and `vol_pred.npy` slice by slice. If one is a rigid transform of the other, it's a coordinate convention bug.
2. **Projection wrong direction**: Render a single test projection and compare to `proj_test_0000.npy`. If they're mirror images, check the `[::-1]` flip in `generate_data.py`.
3. **CUDA extension not found**: `import xray_gaussian_rasterization_voxelization` will fail with ImportError if the CUDA build was not installed in the active conda env. Rebuild from `submodules/`.
4. **TIGRE geometry assertion**: TIGRE will raise if `nVoxel` is in wrong order. Always pass `cfg["nVoxel"][::-1]` (see `ct_utils.py`).
5. **Gaussian count drops to 0**: Set `densify_from_iter` and `densify_until_iter` to high values (beyond training budget) to disable density control entirely. Then confirm the init cloud is non-empty.

---

## 15. Final Internal Summary

### Current Maturity

| Component | Maturity |
|-----------|---------|
| Data preprocessing pipeline (JP2→PNG→npy) | Stable, production-grade |
| TIGRE CT simulation (generate_data.py) | Stable, tested |
| Scanner geometry + scene normalization | Stable, but calibration unverified for Brain2 |
| GaussianModel + training loop | Research-grade, working |
| CUDA rasterizer (2D X-ray projection) | Upstream-quality, stable |
| CUDA voxelizer (3D extraction) | Upstream-quality, stable |
| Initialization (FDK-based) | Research-grade, needs calibration |
| 3D visualization suite | Research-grade, hardcoded paths |
| Evaluation pipeline | Research-grade, functional |
| Traditional baseline comparison | Not executed on Brain2 |
| Format conversion (pickle_to_ply) | Research-grade, has known bug |

### What Must Be Stabilized First

1. Fix the scanner parameter calibration — verify DSD/DSO/sVoxel against actual Brain2 acquisition
2. Fix `debug: false` in training config
3. Run traditional baselines to establish ground truth quality ceiling
4. Fix `pickle_to_ply_v3.py` normalization bug

### What Should Be Rewritten

1. **All `3D_vis_*.py`** — consolidate into one configurable script
2. **pickle conversion** — keep only v3, fix the normalization bug, delete v1/v2
3. **`dummy_*` scripts** — move to `debug/` folder, add docstrings, remove from main path
4. **`PLYcomparer.py`** — remove hardcoded nonexistent path

### What Is Research-Grade vs Production-Grade

**Research-grade (works, not robust):**
- Training loop (no convergence detection, debug mode left on)
- Initialization (density thresholds empirically set, not calibrated)
- Visualization scripts (hardcoded paths throughout)
- Format conversion (has known normalization bug)

**Production-grade (stable, correct):**
- CUDA rasterizer and voxelizer (from upstream R²-Gaussian, well-tested)
- Loss functions (standard implementations)
- TIGRE integration (tested, correct axis handling)
- Data preprocessing (png_stack_to_npy, generate_data) — robust, handles edge cases
- Scene normalization (mathematically correct, applied consistently)

**Not yet executed:**
- Traditional baseline comparison
- Sparse-view ablation (10/25/50 views)
- Real CT data pipeline
- Densification with correct hyperparameters

The core research question — whether Gaussian splatting outperforms classical CT reconstruction on brain data at 75 views — has not been definitively answered. The current 3D PSNR of 6.22 dB likely reflects an experimental configuration problem (scanner parameters, disabled densification) rather than a fundamental limitation of the method. The next team's priority should be resolving the scanner calibration and running traditional baselines before drawing any conclusions.
