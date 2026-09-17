import os
import os.path as osp
import json
import random
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks.loss_functions import calculate_psnr_ldr_torch
from networks.saver import Saver
from networks.pfgl import NelifDecoder
import dataset as dataset_module
from dataset import video_process_tensor_torch, recognization, inverse_data_process_tensor
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
PROJECT_ROOT = osp.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = osp.join(PROJECT_ROOT, "configs", "nelif", "nelif.json")
DEFAULT_CHECKPOINT = osp.join(PROJECT_ROOT, "ckpts", "model.pt")
DEFAULT_OUTPUT = osp.join(PROJECT_ROOT, "outputs", "nelif_test")


# ============================================================
# Args
# ============================================================

def str2checkpoint(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    vl = str(v).lower()
    if vl in ["false", "0", "none", "no", ""]:
        return False
    if vl in ["true", "1", "yes"]:
        return True
    return v


def add_argument():
    parser = argparse.ArgumentParser(
        description="Windows/single-GPU PyTorch test-only version of train_plane_video.py."
    )

    # Required / common
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to json config file")
    parser.add_argument("--ckpt_path", type=str, default=None, help="PyTorch/DeepSpeed checkpoint; defaults to the merged model.pt")

    # Old checkpoint style compatibility:
    # ../ckpts_nelif/{ckpt_name}/{checkpoint_folder}/newest/{epoch_or_newest}/mp_rank_00_model_states.pt
    parser.add_argument("--checkpoint", type=str, default="false")
    parser.add_argument("--ckpt_name", type=str, default="..")
    parser.add_argument("--epoch", default=-1, type=int)

    # Experiment / output
    parser.add_argument("--label", default="", type=str)
    parser.add_argument("--job_name", type=str, default="test")
    parser.add_argument("--save_path", type=str, default="..")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT, help="Directory for test outputs")
    parser.add_argument("--no_save", default=False, action="store_true", help="Only compute metrics, do not save images/exr")

    # Task flags, keep same as old script
    parser.add_argument("--diffuse", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--specular", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--shadow", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--indirect", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--indirect_direct", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--relative", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--voxel", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--tri", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cache", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--cut", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--read_light", default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument("--plane_resolution", type=int, choices=[4, 16, 32, 64, 128], default=None,
                        help="Resize query and position embedding after loading; by default keep the checkpoint resolution")
    parser.add_argument("--light_angular_resolution", default=8, type=int)
    parser.add_argument("--light_direction_resolution", default=128, type=int)

    # Runtime
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", default=0, type=int, help="0 is safest on Windows")
    parser.add_argument("--pin_memory", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--persistent_workers", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--strict", default=True, action=argparse.BooleanOptionalAction,
                        help="Require all current model parameters; known removed legacy modules are reported and excluded")

    args = parser.parse_args()
    args.checkpoint = str2checkpoint(args.checkpoint)
    if args.voxel or args.indirect_direct:
        parser.error("NelifDecoder supports diffuse/specular/shadow/indirect; --voxel and --indirect_direct are not supported")
    if not any((args.diffuse, args.specular, args.shadow, args.indirect)):
        parser.error("Enable at least one rendering branch")
    if not args.read_light:
        parser.error("NelifDecoder.image_encoder requires --read_light")
    if (args.light_angular_resolution, args.light_direction_resolution) != (8, 128):
        parser.error("The current NelifDecoder.image_encoder requires an 8x128 light field")
    if args.ckpt_path is None and args.checkpoint is False:
        args.ckpt_path = DEFAULT_CHECKPOINT
    return args


# ============================================================
# Tensor utilities
# ============================================================

def move_to_device_dtype(obj, device, dtype=torch.float32):
    """Recursive replacement for the old DeepSpeed-side to_cuda_type.

    Float tensors are moved to device and cast to dtype.
    Integer/bool tensors are moved only.
    """
    if torch.is_tensor(obj):
        obj = obj.to(device, non_blocking=True)
        if obj.is_floating_point():
            obj = obj.to(dtype=dtype)
        return obj
    if isinstance(obj, dict):
        return {k: move_to_device_dtype(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device_dtype(v, device, dtype) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device_dtype(v, device, dtype) for v in obj)
    return obj


def to_cuda_type(data, data_type, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return move_to_device_dtype(data, device, data_type)


# ============================================================
# Config helpers
# ============================================================

def create_directory(path):
    os.makedirs(path, exist_ok=True)
    return path


def ensure_config_fields(config):
    config.setdefault("loss_configs", {})
    config["loss_configs"].setdefault("losses", {})
    config.setdefault("saver_configs", {})
    config["saver_configs"].setdefault("resize", None)
    config["saver_configs"].setdefault("max_num_per_page", 100)
    config["saver_configs"].setdefault("save_during_training", {"enable": True, "num": 5})
    config["saver_configs"].setdefault("save_features", {})
    return config


def add_loss_tem(config, name, loss, weight, reweigh=False, relative=False):
    config["loss_configs"]["losses"][name] = {
        "pname": name,
        "gname": name,
        "weight": weight,
        "mask": None,
        "loss": loss,
        "reweigh": reweigh,
        "visualize": True,
        "relative": relative,
    }


def add_save_tem(config, name):
    config["saver_configs"]["save_features"][name] = {
        "visualize_loss": [None],
        "outputs": {"exr": []},
    }


def make_test_saver_config():
    return {
        "saver_configs": {
            "resize": None,
            "max_num_per_page": 100,
            "save_during_training": {"enable": True, "num": 5},
            "save_features": {},
        }
    }


def generate_config_by_args(configs, args):
    configs = ensure_config_fields(configs)
    # Rebuild losses for the enabled outputs rather than keeping training-only losses.
    configs["loss_configs"]["losses"] = {}
    test_config = make_test_saver_config()

    configs["label"] = (args.label or "") + "scene"
    configs.pop("validation_set", None)
    configs.pop("datasets_dir", None)

    # Keep old script's Dataset overrides.
    configs["Dataset"]["light_angular_resolution"] = args.light_angular_resolution
    configs["Dataset"]["light_direction_resolution"] = args.light_direction_resolution
    configs["Dataset"]["indirect"] = args.indirect or args.indirect_direct
    configs["Dataset"]["diffuse"] = args.diffuse
    configs["Dataset"]["specular"] = args.specular
    configs["Dataset"]["shadow"] = args.shadow
    configs["Dataset"].pop("light", None)
    configs["Dataset"].pop("plane_label", None)
    configs["Dataset"]["voxel"] = args.voxel
    configs["Dataset"]["load_tri"] = args.tri
    configs["Dataset"]["cache"] = args.cache
    configs["Dataset"]["cut"] = args.cut
  
    configs["Dataset"]["read_light"] = args.read_light
    configs["Dataset"].pop("datasets_config", None)


    configs["model_configs"]["light_angular_resolution"] = args.light_angular_resolution
    configs["model_configs"]["light_direction_resolution"] = args.light_direction_resolution
    configs["model_configs"].pop("light_path", None)

    if args.diffuse:
        add_loss_tem(configs, "log1p_diffuse_direct_shading", "L1", 1.0, False, False)
        for k in [
            "log1p_diffuse_direct_shading",
            "pred_log1p_diffuse_direct_shading",
            "diffuse_direct_shading",
            "pred_diffuse_direct_shading",
        ]:
            add_save_tem(configs, k)
        for k in ["pred_diffuse_direct_shading", "diffuse_direct_shading"]:
            add_save_tem(test_config, k)

    if args.specular:
        add_loss_tem(configs, "log1p_specular_direct_shading", "L1", 1.0, False, False)
        for k in [
            "log1p_specular_direct_shading",
            "specular_direct_shading",
            "pred_log1p_specular_direct_shading",
            "pred_specular_direct_shading",
        ]:
            add_save_tem(configs, k)
        for k in ["pred_specular_direct_shading", "specular_direct_shading"]:
            add_save_tem(test_config, k)

    if args.shadow:
        add_loss_tem(configs, "shadow", "L1", 1.0, True, False)
        for k in ["shadow", "pred_shadow", "direct_shadow_shading", "pred_direct_shadow_shading"]:
            add_save_tem(configs, k)
        for k in ["pred_shadow", "shadow"]:
            add_save_tem(test_config, k)

    if args.indirect:
        add_loss_tem(configs, "log1p_diffuse_indirect_shading", "L1", 1.0, False, False)
        add_loss_tem(configs, "log1p_specular_indirect_shading", "L1", 1.0, False, False)
        for k in [
            "diffuse_indirect_shading",
            "log1p_diffuse_indirect_shading",
            "specular_indirect_shading",
            "log1p_specular_indirect_shading",
            "pred_log1p_diffuse_indirect_shading",
            "pred_log1p_specular_indirect_shading",
            "pred_specular_indirect_shading",
            "pred_diffuse_indirect_shading",
            "pred_indirect_shading",
            "indirect_shading",
        ]:
            add_save_tem(configs, k)
        for k in ["mask", "pred_indirect_shading", "indirect_shading"]:
            add_save_tem(test_config, k)

    if args.diffuse and args.specular and args.shadow:
        for k in ["direct_shadow_shading", "pred_direct_shadow_shading"]:
            add_save_tem(test_config, k)
        if args.indirect:
            for k in ["shading", "pred_shading"]:
                add_save_tem(test_config, k)


    return configs, test_config


# ============================================================
# Model / checkpoint loading
# ============================================================

def make_model(configs, args):
    model_name = configs["model"]
    if model_name != "NelifDecoder":
        raise ValueError(f"nelif_run supports NelifDecoder, got {model_name!r}")
    return NelifDecoder(
        configs["model_configs"],
        configs["loss_configs"],
        need_direct=bool(args.diffuse or args.specular),
        need_indirect=bool(args.indirect),
        need_shadow=bool(args.shadow),
    )


def resolve_checkpoint_path(path_or_dir):
    """Resolve common DeepSpeed/PyTorch checkpoint locations.

    Accepted:
      1. direct file: mp_rank_00_model_states.pt / *.pt
      2. DeepSpeed tag dir: .../newest/12/
      3. DeepSpeed root dir: .../newest/ containing 'latest'
    """
    if not path_or_dir:
        return ""
    path_or_dir = osp.expanduser(path_or_dir)

    if osp.isfile(path_or_dir):
        return path_or_dir
    if not osp.isdir(path_or_dir):
        return path_or_dir

    for name in ["mp_rank_00_model_states.pt", "model.pt", "latest.pt", "pytorch_model.pt"]:
        p = osp.join(path_or_dir, name)
        if osp.isfile(p):
            return p

    latest_file = osp.join(path_or_dir, "latest")
    if osp.isfile(latest_file):
        with open(latest_file, "r") as f:
            tag = f.read().strip()
        for p in [
            osp.join(path_or_dir, tag, "mp_rank_00_model_states.pt"),
            osp.join(path_or_dir, tag, "model.pt"),
            osp.join(path_or_dir, tag, "latest.pt"),
        ]:
            if osp.isfile(p):
                return p

    matches = []
    for root, _, files in os.walk(path_or_dir):
        if "mp_rank_00_model_states.pt" in files:
            matches.append(osp.join(root, "mp_rank_00_model_states.pt"))
        if "latest.pt" in files:
            matches.append(osp.join(root, "latest.pt"))
        if "model.pt" in files:
            matches.append(osp.join(root, "model.pt"))

    if matches:
        matches.sort(key=lambda p: osp.getmtime(p), reverse=True)
        return matches[0]

    return path_or_dir


def extract_model_state_dict(checkpoint):
    """Extract model weights from DeepSpeed or normal PyTorch checkpoint."""
    if not isinstance(checkpoint, dict):
        return checkpoint

    # DeepSpeed mp_rank_00_model_states.pt usually stores weights under 'module'.
    for key in ["module", "model", "state_dict", "model_state_dict"]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    # Raw state_dict fallback.
    values = list(checkpoint.values())
    if values and sum(torch.is_tensor(v) for v in values) >= max(1, int(0.5 * len(values))):
        return checkpoint

    raise KeyError(f"Cannot find model weights in checkpoint. Top-level keys = {list(checkpoint.keys())[:30]}")


def strip_prefix_candidates(state_dict, model):
    model_keys = set(model.state_dict().keys())
    prefixes = [
        "module.",
        "model.",
        "_forward_module.",
        "_orig_mod.",
        "module.model.",
        "model.module.",
    ]

    candidates = [state_dict]
    for prefix in prefixes:
        candidates.append({
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        })

    # Also try repeatedly stripping common prefixes from every key.
    stripped = {}
    for k, v in state_dict.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
                    changed = True
        stripped[nk] = v
    candidates.append(stripped)

    def score(sd):
        return len(set(sd.keys()) & model_keys)

    best = max(candidates, key=score)
    return best


def load_checkpoint_into_model(model, ckpt_path, strict=False):
    ckpt_path = resolve_checkpoint_path(ckpt_path)
    if not ckpt_path or not osp.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = extract_model_state_dict(checkpoint)
    state_dict = strip_prefix_candidates(state_dict, model)

    current_keys = set(model.state_dict())
    removed_keys = []
    for key in state_dict:
        if key in current_keys:
            continue
        legacy_adaln = (
            key.startswith("image_encoder.layers.encoder_layer_")
            and ".adaLN_modulation." in key
        )
        if legacy_adaln or key.startswith(("upsampler.", "adaptor.", "trioutputlayer.")):
            removed_keys.append(key)
    if removed_keys:
        print(f"Excluding {len(removed_keys)} weights from modules removed in the current pfgl.NelifDecoder:")
        for key in removed_keys:
            print(f"  {key}")
        removed_keys = set(removed_keys)
        state_dict = {k: v for k, v in state_dict.items() if k not in removed_keys}

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded model weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print("Missing keys sample:", missing[:20])
    if unexpected:
        print("Unexpected keys sample:", unexpected[:20])
    return ckpt_path


def configure_plane_resolution(model, resolution=None):
    if resolution is not None:
        model.set_plane_resolution(resolution)
    query_resolution = model.image_to_plane.output_resolution
    position_shape = tuple(model.plane_pos_embedding.shape[-2:])
    if position_shape != (query_resolution, query_resolution):
        raise ValueError(
            f"Checkpoint query resolution {query_resolution} and position embedding {position_shape} differ. "
            "Use --plane_resolution to explicitly resize both."
        )
    if query_resolution not in (4, 16, 32, 64, 128):
        raise ValueError(f"Unsupported sampling resolution: {query_resolution}")
    model.plane_res = query_resolution
    print(f"Triplane resolution: {query_resolution}x{query_resolution}")


def build_old_style_ckpt_path(args, configs):
    ck_epoch = "newest" if args.epoch == -1 else str(args.epoch)
    # If --checkpoint true, use old ablation folder name convention.
    if args.checkpoint is True:
        ck_folder = "{}{}_gpu{}_lr{}_accumulate{}".format(
            args.label or "",
            "scene",
            1,
            0.0003,
            1,
        )
    else:
        ck_folder = str(args.checkpoint)

    return osp.join(
        "..",
        "ckpts_nelif",
        args.ckpt_name,
        ck_folder,
        "newest",
        ck_epoch,
        "mp_rank_00_model_states.pt",
    )


# ============================================================
# Test loop
# ============================================================

def run_test(model, validation_loader, args, device, data_type, test_saver=None, outdir=None):
    model.eval()
    metric_map = {}

    with torch.no_grad():
        pbar = tqdm(validation_loader, desc="Test")
        for sample_idx, (val_data, val_name) in enumerate(pbar):

            print(val_name)
            val_data = to_cuda_type(val_data, data_type, device)
            val_data = video_process_tensor_torch(
                val_data,
                args.indirect or args.indirect_direct,
                args.relative,
                args.voxel,
            )
            val_data["local"] = recognization(val_data["local"])


            val_frame_data, val_frame_preds, val_loss_map, _ = model(
                val_data,
                need_diffuse=args.diffuse,
                need_specular=args.specular,
                need_shadow=args.shadow,
                need_indirect=args.indirect,
                need_volume=False,
                channel_cnt=1 if model.channel_cut else 3,
            )

            inverse_data_process_tensor(
                val_frame_data,
                val_frame_preds,
                args.diffuse,
                args.specular,
                args.shadow,
                args.indirect,
                args.indirect_direct,
                model.channel_cut,
                args.relative,
            )

            for shading_key in [
                "diffuse_direct_shading",
                "specular_direct_shading",
                "direct_shadow_shading",
                "diffuse_indirect_shading",
                "specular_indirect_shading",
                "indirect_shading",
                "shading",
            ]:
                if shading_key not in val_frame_preds:
                    continue

                metric_map.setdefault(shading_key + "_psnr", 0.0)
                metric_map.setdefault(shading_key + "_cnt", 0.0)

                psnr_tensor = calculate_psnr_ldr_torch(
                    val_frame_data["local"][shading_key].detach(),
                    val_frame_preds[shading_key].detach(),
                )
                psnr_value = psnr_tensor.mean()
                if not torch.isinf(psnr_value) and not torch.isnan(psnr_value):
                    metric_map[shading_key + "_psnr"] += psnr_value.item()
                    metric_map[shading_key + "_cnt"] += 1.0

            if (not args.no_save) and test_saver is not None and outdir is not None:
                test_saver.save(
                    val_frame_preds,
                    val_frame_data,
                    val_loss_map,
                    True,
                    osp.join(outdir, "test"),
                    0,
                    val_name,
                )

    if (not args.no_save) and test_saver is not None and outdir is not None:
        # Some Saver implementations buffer pages before writing final html/png pages.
        if hasattr(test_saver, "output_pages"):
            test_saver.output_pages(osp.join(outdir, "test"))

    print("\n--- Test Results ---")
    results = {}
    for psnr_key in sorted(metric_map.keys()):
        if psnr_key.endswith("_cnt"):
            continue
        shading_key = psnr_key[:-5]
        total = metric_map[shading_key + "_psnr"]
        cnt = metric_map[shading_key + "_cnt"]
        if cnt > 0:
            avg = total / cnt
            results[shading_key] = avg
            print(f" * {shading_key}: {avg:.4f} dB")

    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    args = add_argument()

    # dataset.py contains paths relative to src/, so make execution independent
    # of the directory from which PowerShell launches this file.
    args.config = osp.abspath(args.config)
    args.ckpt_path = osp.abspath(args.ckpt_path) if args.ckpt_path else ""
    args.output_dir = osp.abspath(args.output_dir) if args.output_dir else ""
    os.chdir(SCRIPT_DIR)

    if not osp.exists(args.config):
        raise FileNotFoundError(f"Config file does not exist: {args.config}")

    with open(args.config, "r") as f:
        configs = json.load(f)

    seed = int(configs.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    configs, test_config = generate_config_by_args(configs, args)

    if args.output_dir:
        outdir = create_directory(args.output_dir)
    else:
        ablation_name = f"{configs['label']}_pytorch_test_only"
        outdir = create_directory(osp.join(args.save_path, "output_tog", args.job_name, ablation_name))
    os.makedirs(osp.join(outdir, "test"), exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("args.device is cuda, but torch.cuda.is_available() is False")

    data_type = torch.float32

    print(f"Load model {configs['model']}...")
    model = make_model(configs, args)

    ckpt_path = args.ckpt_path
    if not ckpt_path and args.checkpoint is not False:
        ckpt_path = build_old_style_ckpt_path(args, configs)
    if not ckpt_path:
        raise ValueError("Please provide --ckpt_path, or use old --checkpoint/--ckpt_name/--epoch arguments.")

    loaded_path = load_checkpoint_into_model(model, ckpt_path, strict=args.strict)
    print(f"Checkpoint loaded from: {loaded_path}")
    configure_plane_resolution(model, args.plane_resolution)

    model = model.to(device)
    model.eval()

    DatasetCls = getattr(dataset_module, configs["datasets_type"], None)
    if DatasetCls is None:
        raise KeyError(f"Dataset class '{configs['datasets_type']}' was not imported. Check dataset import.")
    validation_dataset = DatasetCls(configs["Dataset"], isTest=True)

    loader_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    validation_loader = DataLoader(validation_dataset, **loader_kwargs)

    test_saver = None
    if not args.no_save:
        test_saver = Saver(osp.join(outdir, "latest"), test_config["saver_configs"])
        test_saver.reset()

    print(f"Test samples: {len(validation_dataset)}")
    print(f"Output dir: {outdir}")

    run_test(model, validation_loader, args, device, data_type, test_saver, outdir)
