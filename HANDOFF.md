# Thesis Full Portable — Handoff Document

**Thesis title:** Orthographic Gaussian Splatting for Volumetric Anatomical Visualization
**Researcher:** Mushfiq
**Archive root:** `C:\Personal_Endeavours\Thesis Full Portable\`
**Date written:** 2026-05-25

---

## 1. What This Archive Is

This folder is the complete, self-contained research archive for a Master's/PhD thesis
investigating 3D Gaussian Splatting (3DGS) as a method for visualising CT brain scan
volumes. The work moved through three distinct experimental phases, each represented by
one sub-folder. A fourth folder (`Thesis/`) existed as the initial starting point but was
deleted after a full MD5 audit confirmed it contained nothing unique.

The thesis topic explores **orthographic rendering of Gaussian splats** as a way to
faithfully reproduce CT-style parallel-projection views of volumetric anatomical data,
as opposed to the perspective rendering used in the original 3DGS papers.

---

## 2. Current Folder Structure

```
Thesis Full Portable\
│
├── Misc\                         ← loose assets (videos, PDFs, screenshots, CSVs)
│
├── Thesis_PreDefence\            ← Phase 1 + 2 (cinematic-gaussians experiments)
│   ├── .git\                     ← wrapper repo tracking the full phase arc
│   ├── .gitignore
│   ├── Brain1_JP2\               ← LADAF 18.048um brain scan (1806 JP2 slices) [~large]
│   ├── Brain2_JP2\               ← LADAF 202.0um brain scan  (811  JP2 slices) [~large]
│   ├── JP2 to PNG converter\
│   │   ├── full_converter.py     ← batch JP2→PNG pipeline (all 1806 slices)
│   │   ├── tif_to_png.py         ← supplementary TIF→PNG converter
│   │   └── venv\                 ← self-resolving venv (patched activate.bat)
│   ├── Napari Viewer(Not a must have)\
│   │   ├── view_3d.py            ← napari-based 3D brain slice viewer
│   │   ├── requirements.txt
│   │   └── brainview\            ← self-resolving venv (patched activate.bat)
│   ├── cinematic-gaussians\      ← Phase 1+2 code (has its own git repo inside)
│   └── 1e523224-...-Copy.png     ← preserved early brain render artifact
│
├── Thesis_Defence\               ← Phase 3 (r2_gaussian CT-native 3DGS)
│   ├── r2_gaussian\              ← git repo (fork of Ruyi-Zha/r2_gaussian)
│   ├── simple-knn\               ← standalone simple-knn build (reference copy)
│   └── gaussian-splatting\       ← vanilla graphdeco-inria/gaussian-splatting clone
│
└── HANDOFF.md
```

---

## 3. Research Progression

### Phase 1 — Initial cinematic-gaussians Exploration (`Thesis/`, now deleted)
- Downloaded and ran [KeKsBoTer/cinematic-gaussians](https://github.com/KeKsBoTer/cinematic-gaussians) (TUM anatomy-focused 3DGS fork).
- First run with LADAF Brain1 scan (18.048um, 1806 slices).
- No code modifications beyond getting it running. Earliest render saved as
  `1e523224-d008-4a29-b283-2d0ad1354feb - Copy.png` in `Thesis_PreDefence/`.
- This phase's state is pinned in the wrapper repo at commit `8c9b1a5` (gitlink → `5392c34`).

### Phase 2 — Brain Data Pipeline & Code Fixes (`Thesis_PreDefence/`)
- Added the JP2→PNG converter pipeline (`full_converter.py`) to process raw LADAF scans.
- Added Napari viewer (`view_3d.py`) for interactive slice inspection.
- Fixed `dataset_readers.py` for explicit xyz/rgb/opacity field handling.
- Fixed `train.py` multiprocessing pickling crash on Windows (replaced inline `lambda`
  with named `collate_identity()` function).
- Added `export_ply.py` (export trained model to PLY) and `make_cameras.py`
  (generate orthographic camera JSON from PNG stack).
- Added `tif_to_png.py` for supplementary data format conversion.
- Multiple `Training_Data_v*/` iterations (increasingly refined COLMAP setups).
- Multiple `Test_Model_v*/` and `Tank_Test_Model_v*/` outputs from training runs.
- Wrapper repo Phase 2 commit: `b5a0cbc`.
- cinematic-gaussians own commit: `c7baa94`.

### Phase 3 — CT-Native Gaussian Splatting (`Thesis_Defence/r2_gaussian/`)
- Switched to [Ruyi-Zha/r2_gaussian](https://github.com/Ruyi-Zha/r2_gaussian), which
  natively ingests cone-beam CT projections via TIGRE (no COLMAP needed).
- Integrated `pytigre 3.0.0` for GPU-accelerated CT forward/backward projection.
- Modified `initialize_pcd.py` and `data_generator/initialize_pcd_all.py` for
  CT brain volume point cloud initialisation.
- Built a suite of 34 visualisation scripts (`3D_vis/`) covering 20+ anatomical datasets:
  abdomen, aneurism, backpack, bonsai, box, carp, chest, engine, foot, head, jaw,
  leg, pancreas, pelvis, teapot — each with a static view and GIF rotation variant.
- Built PLY/pickle conversion pipeline: `ASCII_to_bin.py`, `pickle_to_bin.py`,
  `pickle_to_ply.py` (v1–v3), `png_stack_to_npy.py`.
- Built `PLYcomparer.py` — diff tool for comparing exported Gaussian models
  across experiments.
- `dummy_*.py` scripts — TIGRE/renderer integration stubs used during bring-up.
- SIBR viewer camera snapshot JSONs + PNGs from live evaluation sessions.
- r2_gaussian commit: `bc76c2d`.

### Baseline — Vanilla 3DGS (`Thesis_Defence/gaussian-splatting/`)
- Untouched clone of [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting).
- Serves as the unmodified baseline for comparison.
- No thesis-specific changes. Not under a custom wrapper repo.
- Latest upstream commit at time of clone: `54c035f`.
- Only referenced by `r2_gaussian/PLYcomparer.py` (as a comparison PLY data source).

---

## 4. Git Repositories

### `Thesis_PreDefence/` — wrapper repo
Tracks the *surrounding* folder growth across phases. Uses gitlinks (mode 160000)
to point at the inner cinematic-gaussians repo at specific commits without a full
`.gitmodules` submodule setup (inner commits are local-only, not pushed upstream).

| Commit | Phase | cinematic-gaussians gitlink |
|--------|-------|----------------------------|
| `8c9b1a5` | Phase 1 — Thesis: initial exploration | `5392c34` |
| `b5a0cbc` | Phase 2 — Thesis_PreDefence: pipeline & fixes | `c7baa94` |

Tracked files: `.gitignore`, `cinematic-gaussians` (gitlink), `full_converter.py`,
`tif_to_png.py`, `view_3d.py`, `requirements.txt`, early render PNG.

Gitignored (large data alongside repo):
- `Brain1_JP2/`, `Brain2_JP2/` — raw LADAF scan slices
- `JP2 to PNG converter/Output_Data/`, `Output_PNG/` — converter output
- `JP2 to PNG converter/venv/`, `Napari Viewer.../brainview/` — Python envs
- `cinematic-gaussians/Training_Data*/`, `Test_Model*/`, `Tank_Test_Model*/` — runs
- `cinematic-gaussians/submodules/`, `__pycache__/`, `*.pyd` — build artifacts

### `Thesis_PreDefence/cinematic-gaussians/` — inner repo
Fork of KeKsBoTer/cinematic-gaussians. Thesis-specific changes are in one commit
on top of the upstream history.

| Commit | Description |
|--------|-------------|
| `c7baa94` | Thesis changes: dataset_readers fix, Windows multiprocessing fix, export_ply, make_cameras, UTF-8 env |
| `5392c34` | Last upstream commit before thesis modifications |

### `Thesis_Defence/` — wrapper repo
Mirrors the Thesis_PreDefence wrapper pattern. Uses gitlinks (mode 160000) to pin
all three inner repos at specific commits. No `.gitmodules` file.

| Commit | Gitlinks pinned |
|--------|----------------|
| `27fbef3` | `r2_gaussian` → `bc76c2d`, `gaussian-splatting` → `54c035f`, `simple-knn` → `86710c2` |

Tracked files: `.gitignore`, `r2_gaussian` (gitlink), `gaussian-splatting` (gitlink),
`simple-knn` (gitlink).

Gitignored (large data and build artifacts):
- `r2_gaussian/data/`, `r2_gaussian/output/` — training data and model outputs
- `r2_gaussian/viewers/` — 157 MB pre-compiled SIBR viewer binary
- `r2_gaussian/data_generator/synthetic_dataset/Brain2_PNG/` — 431 MB PNG brain slices
- `r2_gaussian/data_generator/synthetic_dataset/Brain2_PNG.npy` — NumPy volume
- `r2_gaussian/data_generator/synthetic_dataset/cone_ntrain_*/` — CT projection datasets
- `r2_gaussian/spec.txt` — conda spec snapshot
- `gaussian-splatting/Data/` — benchmark training scenes (re-downloadable)
- `gaussian-splatting/output/` — trained baseline model outputs
- `gaussian-splatting/SIBR_viewers/` — compiled viewer

### `Thesis_Defence/r2_gaussian/` — inner repo
Fork of Ruyi-Zha/r2_gaussian. Thesis-specific changes are in one commit on top of
upstream.

| Commit | Description |
|--------|-------------|
| `bc76c2d` | Phase 3 thesis work: all custom scripts, 3D_vis suite, myenv.yml |
| `fc2bcad` | Last upstream commit before thesis modifications |

### `Thesis_Defence/gaussian-splatting/` — upstream baseline
No custom commits. Remote: `https://github.com/graphdeco-inria/gaussian-splatting`

---

## 5. Conda Environments

Three environments are installed in `C:\Users\Mushfiq\miniconda3\envs\`.
Lock files are committed inside each repo.

| Env name | Python | PyTorch | CUDA | Lock file location |
|----------|--------|---------|------|--------------------|
| `cin3dgs` | 3.12 | 2.5.1 | 12.1 | `Thesis_PreDefence/cinematic-gaussians/environment.yml` |
| `r2_gaussian` | 3.9 | 1.12.1+cu118 | 11.8 | `Thesis_Defence/r2_gaussian/myenv.yml` |
| `gaussian_splatting` | — | — | — | `Thesis_Defence/gaussian-splatting/environment.yml` |

Both `environment.yml` files are UTF-8 (converted from Windows UTF-16 exports).
Recreate with: `conda env create -f environment.yml`

### Key packages per env

**cin3dgs** (cinematic-gaussians):
- `pytorch=2.5.1`, `cuda-toolkit=12.1`, `pillow`, `plyfile`, `pyyaml`, `tqdm`
- `opencv-python`, `tensorboard`, `torchvision=0.20.1`
- Submodules (build from source): `diff-gaussian-rasterization`, `simple-knn`, `simple-nn`

**r2_gaussian**:
- `pytorch=1.12.1+cu118`, `cudatoolkit=11.6.2`, `python=3.9`
- `pytigre=3.0.0` (GPU CT reconstruction — TIGRE wrapper)
- `open3d=0.19.0`, `pyvista=0.46.3`, `simpleitk=2.5.2`, `scipy=1.13.1`
- `xray-gaussian-rasterization-voxelization=0.0.0` (built from submodule)
- `simple-knn=0.0.0` (built from submodule)

---

## 6. Brain Scan Data (LADAF)

Two scans from the LADAF (Laboratoire d'Anatomie des Alpes Françaises) dataset,
stored as JP2 (JPEG 2000) slice stacks in `Thesis_PreDefence/`:

| Folder | Prefix | Resolution | Slices | Use |
|--------|--------|------------|--------|-----|
| `Brain1_JP2/` | `18.048um_LADAF-2021-17_brain_VOI-03.2_...` | 18.048 µm | 1806 | Primary training data |
| `Brain2_JP2/` | `202.0um_LADAF-2021-17_brain_complete-organ_...` | 202.0 µm | 811 | Whole-organ overview |

Brain1 (18.048um) was the primary experimental dataset throughout all phases.
The converted PNG versions live in:
- `cinematic-gaussians/Training_Data/images/` (1806 PNGs + COLMAP database.db)
- `r2_gaussian/data_generator/synthetic_dataset/Brain2_PNG/` (431 MB, gitignored)

---

## 7. Virtual Environments (Python venv, not conda)

Two lightweight venvs for the data tools in `Thesis_PreDefence/`:

| Path | Purpose |
|------|---------|
| `JP2 to PNG converter/venv/` | Runs `full_converter.py` and `tif_to_png.py` |
| `Napari Viewer(Not a must have)/brainview/` | Runs `view_3d.py` |

Both `activate.bat` files have been patched to use `%~dp0`-relative paths so they
work regardless of where this archive is moved. No hardcoded absolute paths remain.

---

## 8. Known Issues / TODOs

- **`r2_gaussian/submodules/simple-knn`** — shows as `modified` in `git status`
  due to compiled build artifacts (`.pyd` files, `build/`) inside the submodule.
  This is expected. Do not attempt to commit from inside that submodule unless
  doing actual upstream work.

- **`Thesis_PreDefence/cinematic-gaussians` not pushed** — commit `c7baa94` is local only.
  The upstream remote (`KeKsBoTer/cinematic-gaussians`) does not have this commit,
  so the wrapper repo's gitlink at `c7baa94` is not resolvable by anyone else.
  Push to a personal fork if collaboration or backup is needed.

- **`r2_gaussian` 1 commit ahead of origin** — `bc76c2d` is local only.
  Push with `git push` when ready.

---

## 9. What Was Deleted

### `Thesis/` folder
The original starting-point folder was deleted after a rigorous audit:
- All 1806 JP2 files in `Thesis/cinematic-gaussians/Training_Data/` were confirmed
  identical (5/5 MD5 spot-checks) to `Thesis_PreDefence/Brain1_JP2/`.
- All 115 non-pycache source files in `Thesis/cinematic-gaussians/` existed in
  `Thesis_PreDefence/cinematic-gaussians/`. Only absent items were `__pycache__/*.pyc`
  bytecode (worthless derivatives).
- `Thesis/cinematic-gaussians/Output_Model/` was empty (0 files).
- The git history of `Thesis/cinematic-gaussians/` at the pre-modification state is
  captured as commit `5392c34`, pinned by the wrapper repo's Phase 1 commit `8c9b1a5`.
- One unique file existed: an early brain render PNG (UUID-named, ~1.1 MB).
  A copy was preserved at `Thesis_PreDefence/1e523224-d008-4a29-b283-2d0ad1354feb - Copy.png`.

---

## 10. Quick-Start Reference

```bash
# Activate cinematic-gaussians environment
conda activate cin3dgs
cd "Thesis_PreDefence\cinematic-gaussians"
python train.py --config <your_config>

# Activate r2_gaussian environment
conda activate r2_gaussian
cd "Thesis_Defence\r2_gaussian"
python train.py --config <your_config>

# Convert JP2 brain slices to PNG
cd "Thesis_PreDefence\JP2 to PNG converter"
venv\Scripts\activate
python full_converter.py

# Run a 3D visualisation (r2_gaussian)
conda activate r2_gaussian
cd "Thesis_Defence\r2_gaussian\3D_vis"
python 3D_vis_head.py        # example — one script per dataset
python 3D_vis_head_gif.py    # rotating GIF variant

# View brain slices interactively
cd "Thesis_PreDefence\Napari Viewer(Not a must have)"
brainview\Scripts\activate
python view_3d.py
```
