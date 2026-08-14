import os
import os.path as osp
import json
import random
import argparse
from datetime import datetime
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from networks import *
from networks.loss_functions import calculate_psnr_ldr_torch
from networks.saver import Saver
from networks.pfgl import *
from utils.data_utils import to_cuda, get_frame_data, preprocess_channel_cut
from dataset import *
import os
import shutil
from collections import defaultdict
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


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
        description="Pure PyTorch test-only script. It loads DeepSpeed-trained model weights and runs validation/testing."
    )

    # Required / common
    parser.add_argument("--config", type=str, required=True, help="Path to json config file")
    parser.add_argument("--test_data", type=str, required=True, help="Validation/test dataset name")
    parser.add_argument("--ckpt_path", type=str, default="", help="Direct path or directory of DeepSpeed/PyTorch checkpoint")

    # Old checkpoint style compatibility:
    # ../ckpts_nelif/{ckpt_name}/{checkpoint_folder}/newest/{epoch_or_newest}/mp_rank_00_model_states.pt
    parser.add_argument("--checkpoint", type=str, default="false")
    parser.add_argument("--ckpt_name", type=str, default="..")
    parser.add_argument("--epoch", default=-1, type=int)

    # Experiment / output
    parser.add_argument("--label", default="", type=str)
    parser.add_argument("--job_name", type=str, default="test")
    parser.add_argument("--save_path", type=str, default="..")
    parser.add_argument("--output_dir", type=str, default="", help="If set, save test outputs here directly")
    parser.add_argument("--no_save", default=False, action="store_true", help="Only compute metrics, do not save images/exr")

    # Task flags, keep same as old script
    parser.add_argument("--diffuse", default=False, action="store_true")
    parser.add_argument("--specular", default=False, action="store_true")
    parser.add_argument("--shadow", default=False, action="store_true")
    parser.add_argument("--indirect", default=False, action="store_true")

    parser.add_argument("--plane_label", type=str, default="none")
    parser.add_argument("--light", default="MidLight", type=str)
    parser.add_argument("--light_angular_resolution", default=1, type=int)
    parser.add_argument("--light_direction_resolution", default=1, type=int)

    # Runtime
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin_memory", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--persistent_workers", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--prefetch_factor", default=4, type=int)
    parser.add_argument("--strict", default=False, action="store_true", help="Use strict=True when loading weights")

    args = parser.parse_args()
    args.checkpoint = str2checkpoint(args.checkpoint)
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
    test_config = make_test_saver_config()

    configs["label"] = (args.label or "") + (args.test_data or "")
    configs["validation_set"] = args.test_data

    # Keep old script's Dataset overrides.
    configs["Dataset"]["light_angular_resolution"] = args.light_angular_resolution
    configs["Dataset"]["light_direction_resolution"] = args.light_direction_resolution
    configs["Dataset"]["indirect"] = args.indirect 
    configs["Dataset"]["diffuse"] = args.diffuse
    configs["Dataset"]["specular"] = args.specular
    configs["Dataset"]["shadow"] = args.shadow
    configs["Dataset"]["light"] = args.light
    configs["Dataset"]["plane_label"] = args.plane_label


    configs["model_configs"]["light_angular_resolution"] = args.light_angular_resolution
    configs["model_configs"]["light_direction_resolution"] = args.light_direction_resolution
    configs["model_configs"]["light_path"] = r"../datasets2/{}{}x{}/dir{}x{}.zst".format(
        args.light,
        args.light_angular_resolution,
        args.light_direction_resolution,
        args.light_angular_resolution,
        args.light_direction_resolution,
    )

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
        for k in ["mask", "shading", "pred_shading", "pred_indirect_shading", "indirect_shading"]:
            add_save_tem(test_config, k)


    return configs, test_config


# ============================================================
# Model / checkpoint loading
# ============================================================

def make_model(configs, args):
    model_name = configs["model"]
    if model_name not in globals():
        raise KeyError(f"Model class '{model_name}' was not imported. Check networks import.")
    ModelCls = globals()[model_name]
    return ModelCls(
        configs["model_configs"],
        configs["loss_configs"],
        bool(args.diffuse),
        bool(args.indirect),
        bool(args.shadow),
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
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = extract_model_state_dict(checkpoint)
    state_dict = strip_prefix_candidates(state_dict, model)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded model weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print("Missing keys sample:", missing[:20])
    if unexpected:
        print("Unexpected keys sample:", unexpected[:20])
    return ckpt_path


def build_old_style_ckpt_path(args, configs):
    ck_epoch = "newest" if args.epoch == -1 else str(args.epoch)
    # If --checkpoint true, use old ablation folder name convention.
    if args.checkpoint is True:
        ck_folder = "{}{}_gpu{}_lr{}_accumulate{}".format(
            args.label or "",
            args.test_data or "",
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

def parse_scene_light_from_name(val_name):
    """
    val_name from DataLoader with batch_size=1 is usually a list/tuple like [localID].
    localID format is assumed to be:
        sceneID_lightID_xxx_xxx_xxx
    Your old code used:
        scene, light = val_name[0].split("_")[:2]
        scene = scene[:-7]
    so I keep the same scene grouping rule here.
    """
    if isinstance(val_name, (list, tuple)):
        name = val_name[0]
    else:
        name = val_name

    if isinstance(name, (list, tuple)):
        name = name[0]

    name = str(name)
    stem = osp.splitext(osp.basename(name))[0]
    parts = stem.split("_")

    if len(parts) >= 2:
        scene_raw = parts[0]
        light = parts[1]
    else:
        scene_raw = stem
        light = "unknown_light"

    # Keep your original rule: scene = scene[:-7]
    # This is useful if the last 7 chars encode view/config and should be grouped together.
    scene = scene_raw[:-7] if len(scene_raw) > 7 else scene_raw

    return scene, light, stem


def add_psnr_metric(metric_table, group_key, shading_key, value):
    """
    metric_table[group_key][shading_key] = [sum_psnr, count]
    """
    if group_key not in metric_table:
        metric_table[group_key] = {}
    if shading_key not in metric_table[group_key]:
        metric_table[group_key][shading_key] = [0.0, 0]

    metric_table[group_key][shading_key][0] += float(value)
    metric_table[group_key][shading_key][1] += 1


def write_metric_csv(path, metric_table, group_columns):
    """
    Long-format CSV:
        group columns + shading_key + psnr + count
    """
    os.makedirs(osp.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(group_columns + ["shading_key", "psnr", "count"])

        for group_key in sorted(metric_table.keys(), key=lambda x: str(x)):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            for shading_key in sorted(metric_table[group_key].keys()):
                psnr_sum, cnt = metric_table[group_key][shading_key]
                if cnt <= 0:
                    continue
                writer.writerow(list(group_key) + [shading_key, psnr_sum / cnt, cnt])


def run_test(model, validation_loader, args, device, data_type, test_saver=None, outdir=None):
    model.eval()

    # Overall / grouped metrics
    overall_metrics = {}
    scene_metrics = {}
    light_metrics = {}
    scene_light_metrics = {}

    # Per-sample long table
    sample_rows = []

    shading_keys = [
        "diffuse_direct_shading",
        "specular_direct_shading",
        "direct_shadow_shading",
        "diffuse_indirect_shading",
        "specular_indirect_shading",
        "indirect_shading",
        "shading",
    ]

    with torch.no_grad():
        pbar = tqdm(validation_loader, desc="Test")

        for sample_idx, (val_data, val_name) in enumerate(pbar):
            scene, light, sample_name = parse_scene_light_from_name(val_name)

            val_data = to_cuda_type(val_data, data_type, device)
            val_data = video_process_tensor_torch(
                val_data,
                args.indirect
            )
            val_data["local"] = recognization(val_data["local"])

            val_frame_data, val_frame_preds, val_loss_map, _ = model(
                val_data,
                args.diffuse,
                args.specular,
                args.shadow,
                args.indirect
            )

            inverse_data_process_tensor(
                val_frame_data,
                val_frame_preds,
                args.diffuse,
                args.specular,
                args.shadow,
                args.indirect,
                True
            )

            for shading_key in shading_keys:
                if shading_key not in val_frame_preds:
                    continue
                if shading_key not in val_frame_data["local"]:
                    continue

                psnr_tensor = calculate_psnr_ldr_torch(
                    val_frame_data["local"][shading_key].detach(),
                    val_frame_preds[shading_key].detach(),
                )

                psnr_value = psnr_tensor.mean()

                if torch.isinf(psnr_value) or torch.isnan(psnr_value):
                    continue

                psnr_value = psnr_value.item()

                # 1. overall
                add_psnr_metric(
                    overall_metrics,
                    ("all",),
                    shading_key,
                    psnr_value,
                )

                # 2. by scene
                add_psnr_metric(
                    scene_metrics,
                    (scene,),
                    shading_key,
                    psnr_value,
                )

                # 3. by light
                add_psnr_metric(
                    light_metrics,
                    (light,),
                    shading_key,
                    psnr_value,
                )

                # 4. by scene + light
                add_psnr_metric(
                    scene_light_metrics,
                    (scene, light),
                    shading_key,
                    psnr_value,
                )

                # 5. per sample
                sample_rows.append({
                    "sample_idx": sample_idx,
                    "sample_name": sample_name,
                    "scene": scene,
                    "light": light,
                    "shading_key": shading_key,
                    "psnr": psnr_value,
                })

            pbar.set_postfix(scene=scene, light=light)

            # If you want to save images/exr, uncomment this block.
            # if (not args.no_save) and test_saver is not None and outdir is not None:
            #     test_saver.save(
            #         val_frame_preds,
            #         val_frame_data,
            #         val_loss_map,
            #         True,
            #         osp.join(outdir, "test"),
            #         0,
            #         val_name,
            #     )

    if (not args.no_save) and test_saver is not None and outdir is not None:
        if hasattr(test_saver, "output_pages"):
            test_saver.output_pages(osp.join(outdir, "test"))

    # Save CSV files
    metric_dir = osp.join(outdir, "metrics")
    os.makedirs(metric_dir, exist_ok=True)

    write_metric_csv(
        osp.join(metric_dir, "psnr_overall.csv"),
        overall_metrics,
        ["group"],
    )

    write_metric_csv(
        osp.join(metric_dir, "psnr_by_scene.csv"),
        scene_metrics,
        ["scene"],
    )

    write_metric_csv(
        osp.join(metric_dir, "psnr_by_light.csv"),
        light_metrics,
        ["light"],
    )

    write_metric_csv(
        osp.join(metric_dir, "psnr_by_scene_light.csv"),
        scene_light_metrics,
        ["scene", "light"],
    )

    sample_csv_path = osp.join(metric_dir, "psnr_per_sample.csv")
    with open(sample_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "sample_name",
                "scene",
                "light",
                "shading_key",
                "psnr",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    print("\n--- Test Results: Overall PSNR ---")
    results = {}

    for group_key in sorted(overall_metrics.keys(), key=lambda x: str(x)):
        for shading_key in sorted(overall_metrics[group_key].keys()):
            psnr_sum, cnt = overall_metrics[group_key][shading_key]
            if cnt <= 0:
                continue
            avg = psnr_sum / cnt
            results[shading_key] = avg
            print(f" * {shading_key}: {avg:.4f} dB, count={cnt}")

    print("\n--- Saved PSNR CSV files ---")
    print(f" * {osp.join(metric_dir, 'psnr_overall.csv')}")
    print(f" * {osp.join(metric_dir, 'psnr_by_scene.csv')}")
    print(f" * {osp.join(metric_dir, 'psnr_by_light.csv')}")
    print(f" * {osp.join(metric_dir, 'psnr_by_scene_light.csv')}")
    print(f" * {osp.join(metric_dir, 'psnr_per_sample.csv')}")

    return {
        "overall": overall_metrics,
        "by_scene": scene_metrics,
        "by_light": light_metrics,
        "by_scene_light": scene_light_metrics,
        "per_sample": sample_rows,
    }

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    args = add_argument()

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

    model = model.to(device)
    model.eval()

    DatasetCls = globals().get(configs["datasets_type"])
    if DatasetCls is None:
        raise KeyError(f"Dataset class '{configs['datasets_type']}' was not imported. Check dataset import.")
    validation_dataset = DatasetCls(configs["Dataset"], configs["validation_set"], True)

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
