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

```bash
python src/nelif_run.py --device cuda --output_dir outputs/nelif_test
```

The script uses `configs/nelif/nelif.json` and `ckpts/model.pt`, processes all scenes, saves their results, and reports PSNR.

| Option | Meaning |
| --- | --- |
| `--output_dir PATH` | Result directory; defaults to `outputs/nelif_test` under the project root |
| `--ckpt_path PATH` | Use a different checkpoint |
| `--config PATH` | Use a different model/evaluation JSON configuration |
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
