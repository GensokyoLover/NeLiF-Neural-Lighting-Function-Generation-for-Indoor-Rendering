"""Single-device NeLiF training using the same inputs and losses as nelif_run."""

import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import nelif_run as run


@torch.no_grad()
def calculate_training_psnr(data, predictions, model, args):
    """Measure the training forward pass using nelif_run's LDR PSNR definition."""
    predictions = {name: value.detach() for name, value in predictions.items()}
    run.inverse_data_process_tensor(
        data, predictions, args.diffuse, args.specular, args.shadow, args.indirect,
        args.indirect_direct, model.channel_cut, args.relative,
    )
    metrics = {}
    for name in ("diffuse_direct_shading", "specular_direct_shading", "shadow",
                 "direct_shadow_shading", "diffuse_indirect_shading",
                 "specular_indirect_shading", "indirect_shading", "shading"):
        if name in predictions and name in data["local"]:
            value = run.calculate_psnr_ldr_torch(data["local"][name], predictions[name]).item()
            # Match run_test: average finite samples only, and expose the count.
            metrics[name] = value if math.isfinite(value) else None
    return metrics


def add_argument(argv=None):
    parser = run.build_argument_parser(training=True)
    parser.description = "Train NeLiF with backpropagation and Adam."
    parser.set_defaults(output_dir=os.path.join(run.PROJECT_ROOT, "outputs", "nelif_train"),
                        job_name="train")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="Maximum gradient norm; 0 checks finiteness without clipping")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Stop after this many optimizer steps in total; 0 uses all epochs")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Initialize randomly instead of loading checkpoint weights")
    args = run.validate_arguments(parser, parser.parse_args(argv))
    if args.epochs < 1 or args.max_steps < 0:
        parser.error("--epochs must be positive and --max_steps must be nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if any(not math.isfinite(v) or v < 0 for v in (args.weight_decay, args.grad_clip)):
        parser.error("--weight_decay and --grad_clip must be finite and nonnegative")
    if args.num_workers < 0 or args.prefetch_factor < 1:
        parser.error("--num_workers must be nonnegative and --prefetch_factor positive")
    if not args.output_dir:
        parser.error("--output_dir must not be empty")
    if args.from_scratch and (args.checkpoint is not False or args.ckpt_path != run.DEFAULT_CHECKPOINT):
        parser.error("--from_scratch cannot be combined with an explicit checkpoint")
    return args


def train_step(model, batch, optimizer, args, device):
    """Perform one update and return losses/PSNR from its training forward pass."""
    optimizer.zero_grad(set_to_none=True)
    data = run.to_cuda_type(batch, torch.float32, device)
    data = run.video_process_tensor_torch(
        data, args.indirect or args.indirect_direct, args.relative, args.voxel,
    )
    data["local"] = run.recognization(data["local"])
    frame_data, predictions, loss_map, _ = model(
        data,
        need_diffuse=args.diffuse,
        need_specular=args.specular,
        need_shadow=args.shadow,
        need_indirect=args.indirect,
        need_volume=False,
        channel_cnt=1 if model.channel_cut else 3,
    )
    loss = loss_map["final_loss"] if loss_map is not None else None
    if not torch.is_tensor(loss) or loss.numel() != 1 or not loss.requires_grad:
        raise RuntimeError("Training requires a scalar final_loss connected to model parameters")
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Non-finite training loss; optimizer step was not performed")
    loss.backward()
    # Check before updating so an invalid gradient cannot corrupt saved weights.
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), args.grad_clip or math.inf, error_if_nonfinite=True,
    )
    optimizer.step()
    losses = {name: value.detach().mean().item() for name, value in loss_map.items()}
    psnr = calculate_training_psnr(frame_data, predictions, model, args)
    return losses, psnr


def save_checkpoint(path, model, optimizer, configs, args, epoch, global_step):
    """Save weights in a format accepted by nelif_run.load_checkpoint_into_model."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "module": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": configs,
        "train_args": vars(args),
    }, temporary)
    os.replace(temporary, path)


def run_train(model, train_loader, optimizer, args, device, configs):
    model.train()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        totals = {}
        psnr_totals = {}
        psnr_counts = {}
        batches = 0
        with tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}") as progress:
            for data, _names in progress:
                losses, psnr = train_step(model, data, optimizer, args, device)
                global_step += 1
                batches += 1
                for name, value in losses.items():
                    totals[name] = totals.get(name, 0.0) + value
                for name, value in psnr.items():
                    psnr_totals.setdefault(name, 0.0)
                    psnr_counts.setdefault(name, 0)
                    if value is not None:
                        psnr_totals[name] += value
                        psnr_counts[name] += 1
                progress.set_postfix(loss=f"{losses['final_loss']:.6f}", step=global_step)
                if args.max_steps and global_step >= args.max_steps:
                    break
        if not batches:
            raise RuntimeError("Training loader contains no samples")
        averages = {name: value / batches for name, value in totals.items()}
        average_psnr = {name: value / psnr_counts[name] if psnr_counts[name] else None
                        for name, value in psnr_totals.items()}
        history.append({"epoch": epoch, "global_step": global_step,
                        "batches": batches, "losses": averages,
                        "psnr": average_psnr, "psnr_counts": psnr_counts})
        save_checkpoint(outdir / "latest.pt", model, optimizer, configs, args, epoch, global_step)
        with (outdir / "train_history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
        print(f"\nEpoch {epoch}/{args.epochs}: mean_loss={averages['final_loss']:.6f} "
              f"({batches} samples)", flush=True)
        for name, value in averages.items():
            if name != "final_loss":
                print(f"  mean_loss/{name}: {value:.6f}", flush=True)
        for name, value in average_psnr.items():
            formatted = f"{value:.4f} dB" if value is not None else "N/A (no finite samples)"
            print(f"  mean_psnr/{name}: {formatted} ({psnr_counts[name]}/{batches} samples)", flush=True)
        print(f"Saved {outdir / 'latest.pt'}", flush=True)
        if args.max_steps and global_step >= args.max_steps:
            break
    return history


def main(argv=None):
    args = add_argument(argv)
    args.config = os.path.abspath(args.config)
    args.ckpt_path = os.path.abspath(args.ckpt_path) if args.ckpt_path else ""
    args.output_dir = os.path.abspath(args.output_dir)
    # Keep compatibility with dataset.py's paths relative to src/.
    os.chdir(run.SCRIPT_DIR)
    with open(args.config, encoding="utf-8") as file:
        configs = json.load(file)
    seed = int(configs.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    configs, _ = run.generate_config_by_args(configs, args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device is cuda, but CUDA is unavailable")

    model = run.make_model(configs, args)
    if args.from_scratch:
        print(f"Training from scratch for {args.epochs} epochs (random weights; no checkpoint loaded).",
              flush=True)
    if not args.from_scratch:
        checkpoint = args.ckpt_path or run.build_old_style_ckpt_path(args, configs)
        checkpoint = run.resolve_checkpoint_path(checkpoint)
        if Path(checkpoint).resolve() == (Path(args.output_dir) / "latest.pt").resolve():
            raise ValueError("Choose a new --output_dir to preserve the input checkpoint")
        run.load_checkpoint_into_model(model, checkpoint, strict=args.strict)
    # Resizing replaces Parameters, so it must happen before constructing Adam.
    run.configure_plane_resolution(model, args.plane_resolution)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dataset_class = getattr(run.dataset_module, configs["datasets_type"], None)
    if dataset_class is None:
        raise KeyError(f"Unknown dataset class: {configs['datasets_type']}")
    train_dataset = dataset_class(configs["Dataset"], isTest=False)
    loader_options = dict(batch_size=1, shuffle=True, num_workers=args.num_workers,
                          pin_memory=args.pin_memory and device.type == "cuda",
                          persistent_workers=args.persistent_workers and args.num_workers > 0)
    if args.num_workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_dataset, **loader_options)
    print(f"Train samples: {len(train_dataset)}; device: {device}; lr: {args.lr}")
    return run_train(model, train_loader, optimizer, args, device, configs)


if __name__ == "__main__":
    main()
