# [Thesis] Orthographic Gaussian Splatting for Volumetric Anatomical Visualization
### Adapting Differentiable Radiative Rendering to Sparse-View CT Reconstruction

---

## Abstract

This work investigates the application of **3D Gaussian Splatting** as a differentiable scene representation for **sparse-view CT reconstruction** of human brain volumes. Rather than treating tomographic reconstruction as a classical linear inversion problem, we frame it as an inverse rendering problem: a set of 3D Gaussian primitives is optimized end-to-end so that their simulated X-ray projections match a sparse set of observed cone-beam projections. The rendering model is adapted from RGB radiance-field rendering to physical X-ray attenuation via a custom CUDA rasterizer implementing additive Beer-Lambert accumulation. Reconstruction is evaluated against ground-truth CT volumes on LADAF human brain data using both 2D projection metrics and 3D volumetric PSNR/SSIM. The system additionally introduces a dual-kernel CUDA pipeline separating 2D projection rendering from 3D volumetric extraction, enabling simultaneous projection-space training and volume-space regularization. The project spans three experimental phases: initial exploration of anatomy-targeted Gaussian rendering with COLMAP-based camera calibration, construction of a custom CT preprocessing and orthographic camera pipeline, and full CT-native reconstruction via the R²-Gaussian framework (NeurIPS 2024, Zha et al.) applied to a novel brain dataset.

---

## Research Motivation

### The Sparse-View CT Problem

Computed tomography reconstructs internal anatomy by inverting the X-ray projection operator. Clinical protocols acquire 700–2000 projections at finely-spaced angles — a dose-intensive process whose quality is well understood theoretically. The sparse-view regime, where only 25–75 projections are available, is clinically motivated by two concerns: radiation dose reduction (each projection contributes measurable patient exposure) and computational tractability for interventional settings where real-time reconstruction from C-arm CT is required.

Classical reconstruction algorithms fail systematically as projection count decreases. Filtered backprojection (FDK) produces severe streak artifacts along under-sampled angular frequencies. Iterative methods — SART, CGLS, ASD-POCS, and their variants — improve quality at the cost of hours of computation and a large hyperparameter space with no principled stopping criterion. Total-variation minimization approaches can recover piecewise-smooth volumes but require careful regularization strength tuning and are not differentiable end-to-end with respect to scene parameters.

### Why Differentiable Rendering?

The key insight driving this work is a reframing: sparse-view CT is an instance of **inverse rendering**. Given a physical forward model (the X-ray projection operator), we seek a 3D scene representation whose simulated observations match the measured ones. When both the forward model and the scene representation are differentiable, the problem reduces to gradient-based optimization — the same machinery that drives modern machine learning.

This framing transforms the reconstruction problem into one with direct access to the rich tooling of automatic differentiation and GPU-accelerated optimization. It also admits a class of scene representations — neural radiance fields, Gaussian primitives — that encode powerful implicit geometric priors through their parameterization.

### Why Gaussian Splatting Over NeRF?

Neural Radiance Fields (NeRF, Mildenhall et al., 2020) and their CT variants (NAF, IntratomeNeRF, SAX-NeRF) are the established differentiable approach to tomographic reconstruction. However, NeRF requires querying a neural network at every sample point along every ray — a process that is computationally expensive, difficult to interpret, and slow to converge at high spatial resolutions.

3D Gaussian Splatting (3DGS, Kerbl et al., SIGGRAPH 2023) offers a structurally different solution: an **explicit, geometry-first representation** with no neural network in the rendering path. Each Gaussian primitive has a directly interpretable geometric meaning — position, orientation, scale, and density — and is optimized via closed-form gradients through a tile-based CUDA rasterizer. This results in:

- **Convergence speed:** 5–15 minutes on consumer GPU hardware versus hours for NeRF-based CT methods
- **Physical interpretability:** each Gaussian represents a localized density blob, directly analogous to a CT voxel cluster but with continuous and orientation-aware geometry
- **Adaptive spatial resolution:** the densification and pruning mechanism concentrates Gaussian primitives in anatomically complex regions while discarding them in empty space
- **Exact differentiability:** the tile rasterizer supports full CUDA backward passes, enabling gradient-based optimization of all geometric parameters jointly

The central challenge is that CT projection physics is fundamentally different from RGB radiance rendering. The entire rendering stack must be redesigned for the medical domain.

---

## Key Contributions

### 1. CT-Domain Rendering Adaptation (CUDA)

The standard 3DGS rasterizer is designed for RGB rendering via alpha-compositing along viewing rays. X-ray physics requires a fundamentally different forward model. This work adapts the rendering engine with three core modifications:

**Additive Beer-Lambert accumulation** replaces the standard multiplicative opacity compositing:
```
Standard 3DGS:  rendered = Σ αᵢ · cᵢ · Πⱼ<ᵢ (1 − αⱼ)      [RGB, view-dependent]
X-ray adapted:  rendered = Σ ρᵢ · exp(−½ · rᵀ · Σ⁻¹ᵢ · r)    [attenuation, scalar]
```
where ρᵢ is the Gaussian's density parameter and the exponential term is the EWA-splatted 2D Gaussian footprint. This models the line-integral approximation of Beer-Lambert attenuation through a scene of localized density blobs.

**Single-channel density parameterization** replaces view-dependent RGB spherical harmonics. CT attenuation is view-independent — a material has a fixed linear attenuation coefficient regardless of observation angle. The SH pipeline is entirely removed; each Gaussian carries a single scalar density activated through Softplus to enforce non-negativity.

**Dual projection geometry support:** The rasterizer correctly handles both parallel-beam (orthographic) and cone-beam (perspective) CT scanner geometries, selecting the appropriate Jacobian for the EWA covariance projection in each case.

### 2. Dual-Kernel CUDA Pipeline: Projection + Voxelization

CT reconstruction requires two distinct computational operations on the Gaussian cloud:

- **GaussianRasterizer** (`cuda_rasterizer/`): 2D tile-based X-ray projection for training. Tile size: 16×16 pixels. Gaussians are sorted by depth within each tile; EWA-splatted contributions are accumulated additively.
- **GaussianVoxelizer** (`cuda_voxelizer/`): 3D accumulation into a voxel grid for reconstruction evaluation and volumetric regularization. Thread layout: 8×8×8 voxel blocks. For each voxel, contributions are accumulated from all spatially overlapping Gaussians.

Both kernels implement full CUDA backward passes, enabling gradient flow from 3D reconstruction metrics — including the 3D total variation loss — back through the voxelizer to all Gaussian parameters. This is architecturally significant: it means the system can receive gradient signal from 3D spatial structure, not only from 2D projection agreement.

### 3. Scanner-Aware Camera Parameterization

Standard 3DGS uses COLMAP-estimated camera poses. CT scanners have precisely known geometry encoded in their acquisition parameters. This work derives camera-to-world transforms directly from scanner physics:

```python
# Camera-to-world from scanner geometry
R = R_z(θ) @ R_z(+90°) @ R_x(−90°)
T = [DSO·cos(θ), DSO·sin(θ), 0]
```

where θ is the projection angle and DSO is the source-to-origin distance. The full scanner parameter space — source-detector distance (DSD), source-origin distance (DSO), detector resolution and physical size, voxel spacing, and origin offset — is correctly propagated through the projection matrix construction for both parallel and cone-beam geometries. This eliminates the geometrically incorrect approximation of treating CT slices as photographs, which is the primary failure mode of applying standard 3DGS to CT data.

### 4. Consistent Scene Normalization

Gaussian splatting's CUDA EWA mathematics is numerically sensitive to coordinate scale. Physical CT scanner parameters (distances in meters or millimeters, volumes of tens of centimeters) introduce large-magnitude values that destabilize the splatting arithmetic. This work applies a uniform scene normalization that maps the entire volume of interest into a `[-1,1]³` unit cube:

```python
scene_scale = 2 / max(scanner_cfg["sVoxel"])
```

Crucially, this normalization is applied *consistently* to all spatial quantities (scanner distances, detector dimensions, voxel spacings, origin offsets) and to the projection intensity values. This ensures that Gaussian density, scale bounds, and learning rates are all operating in a consistent normalized space throughout training.

### 5. Physics-Motivated Loss Design

Training optimizes three loss terms operating at complementary spatial scales:

| Term | Domain | Role |
|------|--------|------|
| L1 | 2D projection | Direct data fidelity — projection-space agreement |
| 1 − SSIM | 2D projection | Multi-scale structural coherence across spatial frequencies |
| 3D Total Variation | 3D (random 32³ sub-volume) | Piecewise-smooth prior — CT tissue is locally homogeneous |

The TV loss operates on a randomly sampled 32³ sub-volume at each iteration. This provides 3D spatial regularization — encoding the prior that CT density fields are piecewise-smooth — without requiring a full 256³ volume query at every training step, which would be computationally prohibitive.

### 6. LADAF Brain CT Pipeline

A complete preprocessing pipeline was constructed to bring LADAF (Laboratoire d'Anatomie des Alpes Françaises) human brain CT data into the R²-Gaussian training format:

- **JP2→PNG conversion** for the raw JPEG-2000 LADAF scan archive
- **Isotropic resampling and percentile normalization** (`png_stack_to_npy.py`) — a custom preprocessing script that converts PNG slice stacks to 256³ float32 volumes, handling luminosity grayscale conversion, percentile clipping, and anisotropic voxel spacing correction
- **Synthetic X-ray projection** via TIGRE GPU forward projector (`tigre.Ax`) with realistic Poisson and Gaussian noise modeling
- **FDK-bootstrapped initialization** that samples 50k initial Gaussian positions from high-density voxels in the FDK reconstruction, avoiding random initialization in empty space

### 7. Visualization and Analysis Suite

A comprehensive visualization toolkit was developed for analysis and thesis presentation:

- **30+ anatomy-specific 3D rendering scripts** in `3D_vis/` — marching cubes mesh extraction from Gaussian clouds across 20+ anatomical datasets (brain, head, chest, abdomen, pelvis, and more)
- **Animated rotational GIFs** for publication-quality volumetric visualization
- **Open3D scene viewer** (`scripts/visualize_scene.py`) integrating scanner camera frustums, ground-truth volume mesh, and coordinate frames — critical for verifying geometric consistency between scanner parameters and reconstructed volumes
- **3D Slicer export** (`.nii.gz` via SimpleITK) enabling clinical-grade volumetric inspection and comparison
- **Comparative volume slicer** (`plot_utils.py:show_two_volume`) with interactive slider for ground-truth vs. reconstruction inspection across all three axes

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│ LADAF Brain CT (JPEG-2000 slice stack, 202µm isotropic)                 │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ full_converter.py + png_stack_to_npy.py
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Preprocessed Volume  vol_gt.npy  (256³ float32, normalized [0,1])       │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ TIGRE tigre.Ax()
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Synthetic Projection Dataset                                             │
│  75 × (512×512) cone-beam projections  +  meta_data.json               │
│  Poisson(10000) + Gaussian(σ=10) noise model                           │
└────────────┬───────────────────────────────────────────┬────────────────┘
             │ FDK init                                  │ TIGRE FDK
             ▼                                           ▼
┌────────────────────────┐        ┌──────────────────────────────────────┐
│ init_*.npy             │        │ Traditional Baselines                │
│ 50k×{xyz, density}     │        │ FDK / SART / CGLS / ASD-POCS        │
│ FDK-seeded, [-1,1]³    │        │ (scripts/run_traditional_methods.py) │
└───────────┬────────────┘        └──────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Gaussian Optimization (train.py)                                        │
│                                                                          │
│  GaussianModel: N × {xyz(3), density(1), scale(3), rotation(4)}        │
│                                                                          │
│  Per iteration:                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Camera (angle → c2w via scanner physics)                        │   │
│  │       ↓                                                           │   │
│  │  GaussianRasterizer (CUDA)  →  X-ray projection (512×512)       │   │
│  │       ↓                                                           │   │
│  │  L1 + (1−SSIM) loss  ←  GT projection from proj_train/          │   │
│  │                                                                   │   │
│  │  GaussianVoxelizer (CUDA)  →  32³ density sub-volume            │   │
│  │       ↓                                                           │   │
│  │  3D Total Variation loss                                          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  Adam optimizer  ←  exponential LR decay per parameter group           │
│  Adaptive control: clone / split / prune Gaussians                     │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Evaluation (test.py)                                                    │
│                                                                          │
│  GaussianVoxelizer (CUDA) → 256³ density grid                         │
│  Metrics: PSNR 3D, SSIM 3D (per-axis) vs vol_gt.npy                   │
│  Metrics: PSNR 2D, SSIM 2D (per-projection)                           │
│  Export: vol_pred.nii.gz, vol_pred.npy, slice PNGs                    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Technical Deep Dive

### Rendering — EWA Splatting for X-ray

The projection of a 3D Gaussian onto a 2D detector plane follows the EWA (Elliptical Weighted Average) splatting approximation (Zwicker et al., 2002). For a Gaussian with 3D covariance Σ, center µ, and density ρ, the contribution to detector pixel (u, v) is:

```
contribution = ρ · exp(−½ · [Δu, Δv] · (J Σ' J^T)⁻¹ · [Δu, Δv]^T)
```

where Σ' = T_view · Σ · T_view^T is the view-space covariance, J is the projection Jacobian (identity-like for parallel beam; full perspective Jacobian for cone-beam), and [Δu, Δv] is the displacement from the projected Gaussian center to the pixel.

The final rendered projection is the sum of contributions across all Gaussians visible to that camera — an additive accumulation that approximates the line integral of attenuation in Beer-Lambert X-ray physics. The approximation is exact for infinitesimally thin Gaussians relative to their depth extent and remains qualitatively appropriate for geometrically reasonable configurations.

### Volumetric Extraction — 3D Voxelization

Volume reconstruction requires extracting a 3D density grid from the optimized Gaussian cloud. The GaussianVoxelizer accumulates each Gaussian's contribution into a voxel grid using the same EWA Gaussian kernel in 3D:

```
vol[x,y,z] = Σᵢ ρᵢ · exp(−½ · Δxᵢ^T · Σᵢ⁻¹ · Δxᵢ)
```

where Δxᵢ is the displacement from voxel center to Gaussian center i. The CUDA kernel uses 8×8×8 thread blocks, one block per 512-voxel tile of the 3D grid. This voxelizer is the mechanism through which 3D evaluation metrics and 3D TV loss are computed — critically, it is differentiable, allowing gradient flow from volume-space supervision back to all Gaussian parameters.

### Coordinate System Architecture

Three coordinate systems interact across the pipeline and must be handled consistently:

**World space `[-1,1]³`**: All Gaussian positions, camera transforms, and voxel coordinates live here after scene normalization. Scene normalization maps scanner physical units to this space: `scene_scale = 2/max(sVoxel)`.

**TIGRE space**: TIGRE internally uses `(Z,Y,X)` array ordering (depth-last) and a different axis convention for detector offsets. The `get_geometry_tigre()` function handles all necessary axis reversals and reorderings.

**CUDA GLM space**: The CUDA rasterizer inherits GLM (OpenGL Mathematics) conventions from the original 3DGS codebase — column-major matrices, rotation stored as transpose. `getWorld2View2()` and `getProjectionMatrix()` in `graphics_utils.py` produce the correctly-transposed matrices expected by the CUDA kernels.

### Gaussian Parameterization and Activations

The four learnable attributes per Gaussian use activations that enforce physical constraints:

- **Position** (`_xyz`): unconstrained; bounding box enforced via adaptive pruning
- **Density** (`_density`): activated through `Softplus` to enforce ρ > 0 with smooth gradients near zero
- **Scale** (`_scaling`): activated through `sigmoid · (s_max − s_min) + s_min` when bounds are specified, enforcing `scale ∈ [s_min, s_max]`; this prevents Gaussians from collapsing to points (which would be numerically singular) or expanding to fill the entire volume
- **Rotation** (`_rotation`): stored as quaternion, activated through L2 normalization to the unit sphere

For the Brain2 experiment, scale bounds are set to `[0.001, 1.0]` world units — approximately one voxel at minimum to the full volume at maximum, covering the plausible physical range of anatomical structures.

### Adaptive Density Control

The densification mechanism of 3DGS drives Gaussians toward high-gradient regions and away from over-represented regions. Clone and split operations respond to the accumulated 2D screen-space gradient of Gaussian positions. Prune removes Gaussians with density below threshold, outside the scene bounding box, or excessively large in screen space.

In the primary Brain2 experiment, adaptive density control is disabled by setting both `densify_from_iter` and `densify_until_iter` to 35,000 (beyond the 30,000-iteration training budget). This decision was made after initial experiments showed instability with densification enabled; it represents a significant open research question, as the original R²-Gaussian paper demonstrates that densification is critical for high-quality reconstruction.

---

## Research Challenges

### Ill-Posedness of Sparse-View Projection

The fundamental challenge of sparse-view CT is that 75 projections over 360° are highly underdetermined relative to a 256³ volumetric reconstruction. The Radon transform has a large null space at this projection density: many distinct 3D volumes produce identical 2D projections at 75 angles. A 2D-only loss cannot provide a gradient signal that uniquely identifies the correct 3D structure. This manifests in the experiments as a plateau in 3D metrics while 2D metrics continue to improve — the model finds a projection-consistent but volumetrically incorrect solution.

### Domain Shift: Synthetic Projections vs. Real CT Physics

The training pipeline uses TIGRE-synthesized projections with a simplified noise model (Poisson + additive Gaussian). Real CT acquisitions include beam hardening (polychromatic spectrum effects), scatter, detector point spread function, and ring artifacts from detector element non-uniformity. The gap between synthetic and real projections is a primary obstacle to clinical applicability.

### Initialization Quality for Brain Soft Tissue

FDK reconstruction of brain CT at 75 sparse views produces a severely blurred and streak-corrupted volume. Brain soft tissue has low X-ray contrast relative to bone, making it particularly sensitive to under-sampling artifacts. The FDK-based initialization may provide an insufficiently structured starting point for Gaussian optimization, leaving all 50,000 Gaussians in approximately correct but geometrically unresolved positions.

### Geometry Consistency Between Scanner and Dataset

The scanner parameters (DSD=7m, DSO=5m, sVoxel=2m) are adopted from the R²-Gaussian benchmark dataset defaults without calibration to the LADAF Brain2 acquisition geometry. LADAF scans have precisely known voxel spacings (202µm) and brain dimensions (~160mm in the long axis) — these physical constraints have not been propagated into the scanner YAML. This mismatch is the most likely root cause of the poor 3D reconstruction quality observed.

### Memory and Computational Constraints

The CUDA voxelizer performing a 256³ volume query allocates a full float32 grid (~64MB) on the GPU at each evaluation step. The TV loss compounds this by performing an additional 32³ query per training iteration. At 200,000 Gaussians (the specified maximum), the total VRAM budget approaches the capacity of consumer-grade research GPUs. This constrains the scalability of the approach to higher volume resolutions (512³) without architectural changes.

---

## Experimental Results

### Primary Experiment: Brain2_training (30,000 iterations)

| Metric | Value | Interpretation |
|--------|-------|---------------|
| PSNR 3D | 6.22 dB | Near noise floor — 3D reconstruction is not structurally faithful |
| SSIM 3D | 0.014 | Near-zero structural similarity in 3D volumetric comparison |
| PSNR 2D | 11.00 dB | Moderate — projections are partially reconstructed |
| SSIM 2D | 0.557 | The model fits projection structure to a meaningful degree |

**Training trajectory** (from `Misc/evaluation_metrics.csv`): 2D metrics improve monotonically throughout the 30,000 iterations. 3D metrics plateau at their final values by iteration 1,500 and do not improve thereafter. This trajectory strongly suggests the optimization is finding a manifold of projection-consistent but volumetrically unconstrained solutions — a characteristic failure mode of underdetermined inverse problems with insufficient 3D regularization.

**Qualitative observations:** The Gaussian cloud concentrates in the correct spatial region (inside the skull boundary). The `3D_vis/` visualizations show a broadly anatomically plausible structure at iso-surface level. The per-projection renderings are recognizably brain-shaped. However, fine soft-tissue structures — cortical sulci, ventricles, white/gray matter boundaries — are not resolved.

### Rendering Consistency

The 2D SSIM of 0.557 at 30,000 iterations demonstrates that the model is meaningfully fitting the projection data — not simply collapsing to a uniform cloud. The progression from SSIM2D 0.345 (iteration 1) to 0.557 (iteration 30,000) represents genuine learning in projection space. This indicates the CUDA rasterizer, camera parameterization, and training loop are all functioning correctly; the quality limitation is in 3D reconstruction, not in the projection rendering.

### Current Limitations

1. **No traditional baseline comparison on Brain2**: The `scripts/run_traditional_methods.py` infrastructure exists but has not been executed on the Brain2 dataset. Without FDK/SART/CGLS baseline numbers at the same view count, the R²-Gaussian results cannot be contextualized. This is the most important missing piece.

2. **No clinical validation**: All experiments use TIGRE-synthesized projections. Generalization to real clinical CT acquisition is undemonstrated.

3. **Single dataset**: Experiments are limited to one brain CT volume. Cross-subject generalization, pathology robustness, and acquisition protocol sensitivity are open questions.

4. **Densification disabled**: The primary quality-improving mechanism of Gaussian splatting has not been evaluated on this dataset in a controlled manner.

---

## Research Direction

### Open Problems with Clear Research Value

**Sparse-view quality curve.** The theoretically strongest contribution is a characterization of R²-Gaussian reconstruction quality as a function of projection count (10, 25, 50, 75, 100 views) compared to classical algorithms (FDK, SART, CGLS, ASD-POCS) on the same brain CT dataset. The full baseline infrastructure exists; it requires execution and analysis.

**Scanner parameter calibration.** LADAF data has precisely known acquisition parameters. Calibrating the synthetic scanner YAML to the actual Brain2 physical dimensions and verifying geometric consistency via `scripts/visualize_scene.py` is a prerequisite for any quantitative claims. The current mismatch is a solvable engineering problem.

**Densification dynamics on medical data.** The original R²-Gaussian paper demonstrates that adaptive density control is critical for quality. An ablation study on Brain2 — varying `densify_from_iter`, `densify_until_iter`, `densify_grad_threshold`, and `density_min_threshold` — would be a principled contribution and is executable with the existing infrastructure.

**Initialization sensitivity.** FDK initialization on brain soft tissue is likely suboptimal due to the low-contrast nature of neural tissue at 75 views. Comparing random initialization, FDK initialization at different threshold values, and potentially learned initializations from atlas-based priors would characterize the sensitivity and point toward improvement directions.

**Real vs. synthetic projection gap.** The current system uses TIGRE-synthesized projections with a simplified noise model. Adapting the pipeline to ingest real DICOM CT data — or using DRR (Digitally Reconstructed Radiograph) methods on the LADAF volume — would be a necessary step toward clinical applicability.

### Extension Directions

**Multi-scale Gaussian hierarchy.** The current parameterization uses a single scale-bounded Gaussian cloud. A coarse-to-fine initialization strategy — where large Gaussians capture bulk anatomy and smaller ones refine fine structures — could improve convergence on the ill-posed sparse-view problem.

**Learned initialization from atlas.** CT brain anatomy is highly consistent across subjects. A Gaussian initialization derived from a brain atlas (MNI space, registered to the target) could provide a much stronger geometric prior than FDK, particularly for soft-tissue structures that FDK cannot resolve at sparse view counts.

**Parallel vs. cone-beam rendering comparison.** The CUDA rasterizer supports both geometries. A comparative study on the same dataset under both projection models — with matched noise levels — would be a novel contribution to the CT reconstruction literature.

**Clinically-motivated metrics.** PSNR and SSIM are signal-processing metrics not aligned with radiological diagnostic quality. Evaluating reconstruction quality using task-based metrics — detectability of known anatomical landmarks, Hausdorff distance of cortical surface reconstructions, segmentation accuracy of white/gray matter — would significantly strengthen the clinical relevance of the work.

### Clinical Implications

If differentiable Gaussian splatting can produce diagnostically acceptable reconstructions from 20–50 projections — versus the clinical standard of 700–2000 — the implications for interventional radiology are significant. C-arm CT systems used in neurointerventional procedures (aneurysm coiling, thrombectomy) typically acquire 200–400 projections per rotation, limited by gantry speed and patient breath-hold duration. Reducing this requirement by an order of magnitude while maintaining soft-tissue contrast would be clinically meaningful. The method's explicit geometric representation (no neural network black box) and its convergence speed on consumer GPUs are properties aligned with clinical requirements for interpretability and latency.

---

## Technical Stack

### Rendering & Reconstruction
| Component | Technology | Notes |
|-----------|-----------|-------|
| X-ray projection | Custom CUDA 11.8 tile rasterizer | Modified from graphdeco-inria/gaussian-splatting |
| Volume extraction | Custom CUDA 11.8 voxelizer (8³ thread blocks) | Novel addition, not in original 3DGS |
| CT simulation | TIGRE v2.3 (CERN/University of Bath) | GPU-accelerated, multi-algorithm |
| Covariance math | GLM (OpenGL Mathematics, column-major) | Inherited from 3DGS CUDA |

### Optimization
| Component | Technology |
|-----------|-----------|
| Framework | PyTorch 2.0 + autograd |
| Optimizer | Adam (ε=1e-15) with per-parameter LR groups |
| LR schedule | Exponential decay, independent per attribute |
| Logging | TensorboardX |
| Metrics | PSNR, SSIM (2D projection; 3D per-axis) |

### Visualization & Analysis
| Component | Technology |
|-----------|-----------|
| 3D mesh | Open3D 0.18 + scikit-image marching cubes |
| Volume inspection | matplotlib + interactive Slider widget |
| Medical export | SimpleITK (NIfTI .nii.gz for 3D Slicer) |
| Animation | matplotlib multi-angle savefig loop |

### Data Pipeline
| Component | Technology |
|-----------|-----------|
| CT format | JPEG-2000 (LADAF), DICOM (planned), NumPy .npy |
| Slice conversion | glymur (JP2), Pillow, scikit-image |
| Resampling | scipy.ndimage.zoom + skimage.transform.resize |

---

## Repository Structure

```
Thesis/
├── HANDOFF.md                          # Archive-level audit and research progression
├── INTERNAL_HANDOFF.md                 # Full technical handoff for incoming team
├── README.md                           # This document
│
├── Misc/
│   ├── evaluation_metrics.csv          # Training curve: PSNR/SSIM over 30k iterations
│   ├── GaSpCT_*.pdf                    # Reference paper: GaSpCT (Gaussian Splatting for CT)
│   ├── Gaussian Splatting for Cinematic Anatomy.pdf
│   ├── Multi-Layer Gaussian Splatting for Immersive Anatomy Visualization.pdf
│   └── SWE_Orthographic Gaussian Splatting_*.pdf/pptx
│
├── Thesis_PreDefence/                  # Phase 1+2: anatomy-targeted 3DGS + orthographic pipeline
│   ├── JP2 to PNG converter/           # LADAF JP2→PNG preprocessing
│   ├── Napari Viewer/                  # Interactive 3D slice viewer
│   └── cinematic-gaussians/            # KeKsBoTer fork with thesis-specific modifications
│       ├── dataset_readers.py          # Fixed: explicit field handling
│       ├── train.py                    # Fixed: Windows multiprocessing
│       ├── export_ply.py               # Custom: model export
│       └── make_cameras.py             # Custom: orthographic camera generation
│
└── Thesis_Defence/                     # Phase 3: R²-Gaussian CT-native reconstruction [PRIMARY]
    ├── r2_gaussian/                    # Main research codebase
    │   ├── train.py                    # Training entry point
    │   ├── test.py                     # Evaluation
    │   ├── initialize_pcd.py           # FDK-bootstrapped initialization
    │   ├── png_stack_to_npy.py         # Custom volume preprocessing (thesis contribution)
    │   ├── r2_gaussian/                # Core library
    │   │   ├── gaussian/               # Scene representation + rendering
    │   │   │   ├── gaussian_model.py   # GaussianModel (4-param per Gaussian)
    │   │   │   └── render_query.py     # render() + query() wrappers
    │   │   ├── dataset/                # Data loading + CT camera
    │   │   ├── utils/                  # Losses, metrics, CT bridge, visualization
    │   │   └── submodules/             # CUDA extensions
    │   │       ├── xray-gaussian-rasterization-voxelization/
    │   │       └── simple-knn/
    │   ├── data_generator/             # TIGRE-based synthetic CT generation
    │   ├── scripts/                    # Batch training, baselines, format conversion
    │   └── 3D_vis/                     # Anatomy-specific visualization suite (30+ scripts)
    └── gaussian-splatting/             # Upstream 3DGS baseline (unmodified reference)
```

---

## Reproducibility

### Prerequisites

- NVIDIA GPU with ≥16GB VRAM (RTX 3090 recommended)
- CUDA 11.8 (system installation)
- Anaconda or Miniconda
- MSVC (Windows) or GCC (Linux) for CUDA extension compilation

### Environment Setup

```bash
# Create environment
conda env create --file Thesis_Defence/r2_gaussian/environment.yml
conda activate r2_gaussian_new

# Install PyTorch (not in yml — CUDA version must match system)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Windows: set MSVC SDK before building extensions
SET DISTUTILS_USE_SDK=1

# Build CUDA extensions
pip install -e Thesis_Defence/r2_gaussian/r2_gaussian/submodules/xray-gaussian-rasterization-voxelization --no-build-isolation
pip install -e Thesis_Defence/r2_gaussian/r2_gaussian/submodules/simple-knn --no-build-isolation

# Install TIGRE CT simulation toolkit
pip install TIGRE-2.3/Python --no-build-isolation
python Thesis_Defence/r2_gaussian/dummy_TIGRE_run.py  # verify
```

### Data Preparation

```bash
cd Thesis_Defence/r2_gaussian

# Step 1: Volume preprocessing
python png_stack_to_npy.py Brain2_PNG/ vol_gt.npy 256

# Step 2: Synthetic CT projection generation
python data_generator/synthetic_dataset/generate_data.py \
    --vol vol_gt.npy \
    --scanner data_generator/synthetic_dataset/scanner/cone_beam.yml \
    --output data/cone_ntrain_75_angle_360 \
    --n_train 75 --n_test 100

# Step 3: FDK initialization
python initialize_pcd.py \
    --data data/cone_ntrain_75_angle_360/Brain2_PNG_cone \
    --recon_method fdk --n_points 50000
```

### Training

```bash
python train.py \
    -s data/cone_ntrain_75_angle_360/Brain2_PNG_cone \
    -m output/experiment_name \
    --iterations 30000 \
    --lambda_tv 0.05 \
    --lambda_dssim 0.2

# Evaluate
python test.py -m output/experiment_name
```

### Scene Geometry Verification

Before any experiment, verify that camera frustums correctly enclose the reconstructed volume:

```bash
python scripts/visualize_scene.py \
    -s data/cone_ntrain_75_angle_360/Brain2_PNG_cone
```

This renders an Open3D window showing the marching-cubes volume mesh and camera frustums. If frustums do not pass through the volume, scanner parameters require recalibration.

---

## Citation

This project is built on R²-Gaussian (NeurIPS 2024):

```bibtex
@inproceedings{r2gaussian2024,
  title     = {R$^2$-Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction},
  author    = {Ruyi Zha and Tao Jun Lin and Yuanhao Cai and Jiwen Cao and Yanhao Zhang and Hongdong Li},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024}
}
```

3D Gaussian Splatting:

```bibtex
@Article{kerbl3Dgaussians,
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal = {ACM Transactions on Graphics},
  volume  = {42}, number = {4}, year = {2023}
}
```

CT simulation via TIGRE:

```bibtex
@article{biguri2016tigre,
  title   = {TIGRE: a MATLAB-GPU toolbox for CBCT image reconstruction},
  author  = {Biguri, Ander and Dosanjh, Manjit and Hancock, Steven and Soleimani, Manuchehr},
  journal = {Biomedical Physics \& Engineering Express},
  volume  = {2}, number = {5}, year = {2016}
}
```

---

## License

This project inherits the non-commercial research license from 3D Gaussian Splatting (graphdeco-inria). TIGRE is licensed under its own BSD-style terms. LADAF data is used under the conditions of the LADAF data sharing agreement.
