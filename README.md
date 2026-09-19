# NeLiF: Neural Lighting Function Generation for Indoor Rendering

Code and project page for **NeLiF** (SIGGRAPH Asia 2025).

[Paper](https://dl.acm.org/doi/10.1145/3757377.3763958) · [Project page](https://GensokyoLover.github.io/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering/) · [Data and pretrained model](https://huggingface.co/datasets/GensokyoLOvEr/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering/tree/main)

## 1. Clone the code

```bash
git clone https://github.com/GensokyoLover/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering.git
cd NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering
```

Run the following commands from the repository root.

## 2. Install with environment.yml

Create and activate the environment using [environment.yml](environment.yml) (Python 3.10, PyTorch 2.6.0, CUDA 12.6):

```bash
conda env create -f environment.yml
conda activate nelif
```

## 3. Download the scenes, lights, and model

Download these files from [Hugging Face](https://huggingface.co/datasets/GensokyoLOvEr/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering/tree/main):

| Download | Final local location |
| --- | --- |
| [scene.tar.gz](https://huggingface.co/datasets/GensokyoLOvEr/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering/resolve/main/scene.tar.gz?download=true) |  Extracted scene `.pkl.zst` files in `datasets/scene/` |
| [light.zip](https://huggingface.co/datasets/GensokyoLOvEr/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering/resolve/main/light.zip?download=true) | Extracted light `.pkl.zst` files in `datasets/Light/` |
| [model.pt](https://huggingface.co/datasets/GensokyoLOvEr/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering/resolve/main/model.pt?download=true) |  `ckpts/model.pt` |

`datasets/OutDir.exr`, `datasets/indirect_dir.exr`, and `datasets/bias_info.json` are included in Git.

Download using the links above or the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
python -m pip install -U huggingface_hub
hf download GensokyoLOvEr/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering scene.tar.gz light.zip --repo-type dataset --local-dir downloads
hf download GensokyoLOvEr/NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering model.pt --repo-type dataset --local-dir ckpts
```

Extract the archives:

```bash
python -m tarfile -e downloads/scene.tar.gz downloads/scene_unpacked
python -m zipfile -e downloads/light.zip downloads/light_unpacked
```

Place the extracted `.pkl.zst` files directly in `datasets/scene/` and `datasets/Light/`, preserving their file names and the directory capitalization shown below.

## 4. Required directory structure

```text
NeLiF-Neural-Lighting-Function-Generation-for-Indoor-Rendering/
├── README.md
├── configs/
│   └── nelif/
│       └── nelif.json                 # Model/evaluation configuration (Git)
├── ckpts/
│   └── model.pt                       # Download from Hugging Face
├── datasets/
│   ├── OutDir.exr                     # Included in Git
│   ├── indirect_dir.exr               # Included in Git
│   ├── bias_info.json                 # Included in Git
│   ├── scene/
│   │   └── <scene_sample>.pkl.zst      # Extract scene.tar.gz here
│   └── Light/
│       └── <lightID>.pkl.zst           # Extract light.zip here
├── src/
│   ├── nelif_run.py                   # Inference entry point
│   ├── dataset.py
│   ├── common/
│   ├── networks/
│   └── utils/
├── docs/
│   └── index.md                       # Paper project page
└── outputs/                           # Created when running inference
```

## 5. Run inference

Run from the repository root:

```bash
python src/nelif_run.py --config configs/nelif/nelif.json --ckpt_path ckpts/model.pt --label base --job_name base --light_angular_resolution 8 --light_direction_resolution 128 --diffuse --specular --shadow --indirect --device cuda --output_dir outputs/nelif_test
```

This runs all four rendering branches for every scene, saves their results, and reports PSNR.

| Option | Meaning |
| --- | --- |
| `--output_dir PATH` | Result directory; defaults to `outputs/nelif_test` under the project root |
| `--ckpt_path PATH` | Use a different checkpoint |
| `--config PATH` | Use a different model/evaluation JSON configuration |
| `--label base --job_name base` | Evaluation label and job name |
| `--light_angular_resolution 8 --light_direction_resolution 128` | Light-field resolution |
| `--diffuse --specular --shadow --indirect` | Enable all four rendering branches |
| `--device cuda` | Run inference on the GPU |
| `--no_save` | Compute metrics without saving images |

Run `python src/nelif_run.py --help` for all options.

## 6. Find the saved results

Each scene has its own output folder:

```text
outputs/nelif_test/test/data/00000_<scene_sample>/
├── pred_shading_<N>.exr              # Combined predicted shading
├── shading_<N>.exr                   # Corresponding ground truth
├── pred_diffuse_direct_shading_<N>.exr
├── pred_specular_direct_shading_<N>.exr
├── pred_shadow_<N>.exr
├── pred_direct_shadow_shading_<N>.exr
├── pred_indirect_shading_<N>.exr
└── ...                              # Ground-truth components and mask
```

`<N>` is the sample counter, starting at 1. Results are saved as HDR EXR images.

## 7. Train or fine-tune

`nelif_train.py` reuses the inference entry point's model, preprocessing, branch flags,
and L1 losses. It enables training mode and performs `zero_grad`, `backward`, and an
Adam optimizer update for each sample. No additional Python packages are required.

Run from the repository root to fine-tune the released checkpoint:

```bash
python src/nelif_train.py --ckpt_path ckpts/model.pt --epochs 10 --lr 0.0003 --device cuda --output_dir outputs/nelif_train
```

All four rendering branches are enabled by default. Training uses shuffled samples
from `datasets/scene/` and the matching lights in `datasets/Light/`, with batch size 1.
This is the same dataset used by inference; no separate validation split is created.

| Option | Meaning |
| --- | --- |
| `--epochs N` | Number of training epochs; default 1 |
| `--lr RATE` | Adam learning rate; default 0.0003 |
| `--weight_decay VALUE` | Adam weight decay; default 0 |
| `--grad_clip VALUE` | Maximum gradient norm; default 0 disables clipping |
| `--max_steps N` | Stop after N optimizer updates in total; default 0 has no step limit |
| `--from_scratch` | Use random weights instead of loading `ckpts/model.pt` |
| `--plane_resolution N` | Resize the learned planes before creating the optimizer |
| `--no-diffuse`, `--no-specular`, `--no-shadow`, `--no-indirect` | Disable individual branches; keep at least one enabled |

Each epoch (or the final partial epoch when `--max_steps` is reached) saves
`latest.pt` and `train_history.json` under `--output_dir`. The checkpoint includes
model weights, Adam state, configuration, epoch, and step count. `--ckpt_path`
initializes model weights only; it does not resume optimizer state or epoch counters.
Use a new output directory for each run to preserve previous training results.
Every epoch prints the mean total loss, each component loss, and mean PSNR for
each available shading component (including combined `shading` when all branches
are enabled). PSNR uses the same LDR conversion and calculation as `nelif_run.py`.
These are averages of the training forward passes before their optimizer updates,
not a separate validation pass using the final epoch weights. Non-finite PSNR
samples are excluded, as in inference, and valid sample counts are reported;
if none are valid, the metric is printed as N/A and saved as JSON null.
`train_history.json` records `losses`, `psnr`, and `psnr_counts` for each epoch.
Image export remains in `nelif_run.py`.

Train from random initialization for 1000 epochs:

```bash
python src/nelif_train.py --from_scratch --epochs 1000 --lr 0.0003 --device cuda --output_dir outputs/nelif_train_scratch_1000
```

Test a single training step:

```bash
python src/nelif_train.py --max_steps 1 --device cuda --output_dir outputs/nelif_train_check
```

Run inference with the trained weights:

```bash
python src/nelif_run.py --ckpt_path outputs/nelif_train/latest.pt --device cuda --output_dir outputs/nelif_trained_test
```
