"""
train_lora.py — LoRA finetuning entry point for Qwen3-VL-8B-Instruct.

Applies LoRA adapters to attention projection layers in BOTH the vision
encoder and the language model, then runs supervised finetuning using
HuggingFace SFTTrainer (trl).

Usage:
    python vlmrl/vlm/finetune/train_lora.py \
        --train_json models/vlm_lora/<run>/train_split.json \
        --val_json   models/vlm_lora/<run>/val_split.json \
        --output_dir models/vlm_lora/<run>

Output directory structure after training:
    <output_dir>/
    ├── checkpoint-epoch1/
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    ├── checkpoint-epoch2/
    ├── checkpoint-epoch3/
    ├── best/
    ├── training_args.json
    └── training_log.jsonl
"""
import argparse
import json
import os
import shutil
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA finetuning of Qwen3-VL-8B-Instruct for egocentric bbox detection."
    )
    # Required paths
    parser.add_argument("--train_json", type=str, required=True,
                        help="Path to train_split.json produced by prepare_dataset.py")
    parser.add_argument("--val_json", type=str, required=True,
                        help="Path to val_split.json produced by prepare_dataset.py")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save checkpoints and logs (e.g., models/vlm_lora/<run>)")

    # Model
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="HuggingFace model name or path (default: Qwen/Qwen3-VL-8B-Instruct)")
    parser.add_argument("--freeze_vision_encoder", action="store_true",
                        help="Freeze LoRA adapters in the vision encoder; only LM LoRA layers are trained.")

    # LoRA hyperparameters
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank r (default: 32)")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha scaling factor (default: 64)")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout probability (default: 0.05)")

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per-device training batch size (default: 4)")
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps, default: 4)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for LR scheduler (default: 0.1)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (default: 1.0)")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length (default: 2048)")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cuda:0' etc. (default: cuda)")

    return parser.parse_args()


def load_base_model(base_model_name: str, device: str):
    """
    Load Qwen3-VL-8B-Instruct in bfloat16 with all weights frozen.

    Returns:
        (model, processor)
    """
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"Loading processor from: {base_model_name}")
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)

    print(f"Loading base model onto {device} in bfloat16...")
    is_cuda = device == "cuda" or device.startswith("cuda:")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if is_cuda else torch.float32,
        device_map={"": device} if is_cuda else "cpu",
        trust_remote_code=True,
    )

    # Freeze all base weights — only LoRA adapter params will be trainable
    model.requires_grad_(False)
    print("Base model loaded. All weights frozen.")

    return model, processor


def apply_lora(model, args):
    """
    Apply LoRA adapters to attention projection layers in both the vision
    encoder (model.visual) and the language model (model.model).

    Collects target module names by inspecting model.named_modules() for
    q_proj, k_proj, v_proj, o_proj in both namespaces.

    Returns:
        peft_model
    """
    from peft import LoraConfig, TaskType, get_peft_model

    # Collect all module names matching projection layers in both vision encoder and LM
    target_modules = set()
    for name, module in model.named_modules():
        if any(proj in name for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            # Extract the leaf module name (last component)
            leaf = name.split(".")[-1]
            target_modules.add(leaf)

    if not target_modules:
        print(
            "WARNING: No q_proj/k_proj/v_proj/o_proj modules found via inspection. "
            "Falling back to hardcoded target_modules list.",
            file=sys.stderr,
        )
        target_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}

    target_modules_list = sorted(target_modules)
    print(f"LoRA target_modules: {target_modules_list}")

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules_list,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, lora_config)
    return peft_model


def freeze_vision_encoder_lora(model) -> None:
    """
    Freeze all LoRA adapter parameters that belong to the vision encoder.
    Called after apply_lora() when --freeze_vision_encoder is set.
    """
    frozen = 0
    for name, param in model.named_parameters():
        if param.requires_grad and "visual" in name:
            param.requires_grad_(False)
            frozen += param.numel()
    print(f"Vision encoder LoRA frozen: {frozen:,} parameters set to requires_grad=False")


def print_trainable_params(model) -> None:
    """Print total, trainable, vision encoder, and LM trainable parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ve_trainable = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and "base_model.model.visual" in name
    )
    lm_trainable = trainable_params - ve_trainable

    print(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    print(f"  Vision encoder trainable: {ve_trainable:,}")
    print(f"  LM trainable:             {lm_trainable:,}")


def rename_checkpoints_to_epoch_names(output_dir: str, n_epochs: int) -> None:
    """
    Rename Trainer's auto-named checkpoints (checkpoint-<step>) to
    checkpoint-epoch1, checkpoint-epoch2, etc. by sorting by step number.
    """
    import re

    checkpoint_dirs = []
    for name in os.listdir(output_dir):
        full_path = os.path.join(output_dir, name)
        if os.path.isdir(full_path) and re.match(r"^checkpoint-\d+$", name):
            step = int(name.split("-")[1])
            checkpoint_dirs.append((step, full_path, name))

    checkpoint_dirs.sort(key=lambda x: x[0])

    for epoch_idx, (step, full_path, name) in enumerate(checkpoint_dirs, start=1):
        new_name = f"checkpoint-epoch{epoch_idx}"
        new_path = os.path.join(output_dir, new_name)
        if not os.path.exists(new_path):
            os.rename(full_path, new_path)
            print(f"  Renamed {name} -> {new_name}")
        else:
            print(f"  {new_name} already exists; skipping rename of {name}")


def save_best_checkpoint(trainer, output_dir: str) -> None:
    """
    Copy the trainer's best model checkpoint to <output_dir>/best/.
    """
    best_ckpt = trainer.state.best_model_checkpoint
    if best_ckpt is None:
        print("WARNING: No best checkpoint found in trainer state.", file=sys.stderr)
        return

    best_dest = os.path.join(output_dir, "best")
    if os.path.exists(best_dest):
        shutil.rmtree(best_dest)
    shutil.copytree(best_ckpt, best_dest)
    print(f"Best checkpoint copied to: {best_dest}")


def save_training_log(trainer, output_dir: str) -> None:
    """Write training log history to training_log.jsonl."""
    log_path = os.path.join(output_dir, "training_log.jsonl")
    with open(log_path, "w") as f:
        for entry in trainer.state.log_history:
            f.write(json.dumps(entry) + "\n")
    print(f"Training log saved to: {log_path}")


class IouSFTTrainer:
    """
    Mixin / factory that wraps SFTTrainer so that evaluate() augments the
    standard eval-loss metrics with generation-based IoU detection metrics
    (mean_iou, precision, recall, F1, json_parse_rate).

    The metric ``eval_mean_iou`` is returned by evaluate() so that the
    Trainer's built-in best-checkpoint logic (load_best_model_at_end) can
    use it directly when metric_for_best_model="eval_mean_iou".
    """

    # Set by build_iou_sft_trainer() before training.
    _iou_val_samples: list = []
    _iou_processor = None
    _iou_device: str = "cuda"
    _iou_max_new_tokens: int = 256

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        import torch
        from vlmrl.vlm.finetune.eval_lora import (
            compute_metrics as compute_iou_metrics,
            run_inference,
        )

        # Standard loss-based eval from SFTTrainer
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        if not self._iou_val_samples or self._iou_processor is None:
            return metrics

        # Generation-based IoU eval on the raw validation samples
        print("\n[IouSFTTrainer] Running generation-based IoU evaluation...")
        self.model.eval()
        with torch.no_grad():
            results = run_inference(
                self.model,
                self._iou_processor,
                self._iou_val_samples,
                self._iou_device,
                self._iou_max_new_tokens,
            )

        iou_metrics = compute_iou_metrics(results)
        overall = iou_metrics["overall"]

        tp = overall["tp"]
        fp = overall["fp"]
        fn = overall["fn"]
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        # Inject into the metrics dict so _save_checkpoint sees them
        prefix = metric_key_prefix
        metrics[f"{prefix}_mean_iou"] = overall["mean_iou"]
        metrics[f"{prefix}_precision"] = overall["precision"]
        metrics[f"{prefix}_recall"] = overall["recall"]
        metrics[f"{prefix}_f1"] = f1
        metrics[f"{prefix}_json_parse_rate"] = iou_metrics["json_parse_rate"]
        metrics[f"{prefix}_tp"] = tp
        metrics[f"{prefix}_fp"] = fp
        metrics[f"{prefix}_fn"] = fn

        print(
            f"[IouSFTTrainer] mean_iou={overall['mean_iou']:.4f}  "
            f"precision={overall['precision']:.4f}  recall={overall['recall']:.4f}  "
            f"F1={f1:.4f}  parse_rate={iou_metrics['json_parse_rate']:.3f}"
        )

        # Re-log so the augmented metrics appear in training_log.jsonl
        self.log(metrics)

        return metrics


def build_iou_sft_trainer(SFTTrainerClass, val_samples_raw, processor, device, max_new_tokens):
    """
    Dynamically create an IouSFTTrainer subclass from SFTTrainerClass so that
    the IoU evaluate() override is mixed in regardless of which trl version is
    installed (SFTTrainer is defined at import time).
    """
    cls = type(
        "IouSFTTrainer",
        (IouSFTTrainer, SFTTrainerClass),
        {},
    )
    cls._iou_val_samples = val_samples_raw
    cls._iou_processor = processor
    cls._iou_device = device
    cls._iou_max_new_tokens = max_new_tokens
    return cls


def main():
    args = parse_args()

    # Validate dependencies early
    try:
        import peft  # noqa
    except ImportError:
        print(
            "ERROR: peft is required. Install with: pip install peft>=0.12.0",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        import trl  # noqa
    except ImportError:
        print(
            "ERROR: trl is required. Install with: pip install trl>=0.12.0",
            file=sys.stderr,
        )
        sys.exit(1)

    import torch
    from trl import SFTConfig, SFTTrainer

    from vlmrl.vlm.finetune.dataset import LoRAFinetuneDataset, collate_fn

    os.makedirs(args.output_dir, exist_ok=True)

    # Save training args for reproducibility
    training_args_path = os.path.join(args.output_dir, "training_args.json")
    with open(training_args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Training args saved to: {training_args_path}")

    print("\n" + "=" * 60)
    print("LoRA Finetuning — Qwen3-VL-8B-Instruct")
    print("=" * 60)
    print(f"Base model:     {args.base_model}")
    print(f"LoRA rank:      {args.lora_rank}")
    print(f"LoRA alpha:     {args.lora_alpha}")
    print(f"LoRA dropout:   {args.lora_dropout}")
    print(f"Freeze VE LoRA: {args.freeze_vision_encoder}")
    print(f"Learning rate:  {args.lr}")
    print(f"Batch size:     {args.batch_size} (x{args.grad_accum_steps} grad accum = {args.batch_size * args.grad_accum_steps} effective)")
    print(f"Epochs:         {args.epochs}")
    print(f"Output dir:     {args.output_dir}")
    print("=" * 60 + "\n")

    # Load base model and apply LoRA
    model, processor = load_base_model(args.base_model, args.device)
    peft_model = apply_lora(model, args)
    peft_model.config.use_cache = False  # Required for gradient checkpointing compatibility

    if args.freeze_vision_encoder:
        freeze_vision_encoder_lora(peft_model)

    print("\nTrainable parameter summary:")
    print_trainable_params(peft_model)
    print()

    # Build datasets
    print("Loading datasets...")
    train_dataset = LoRAFinetuneDataset(args.train_json, processor, args.max_seq_length)
    val_dataset = LoRAFinetuneDataset(args.val_json, processor, args.max_seq_length)
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")

    # Configure SFTTrainer
    # Best checkpoint is selected by eval_mean_iou (generation-based IoU evaluation),
    # falling back to eval_loss if IoU metrics are unavailable.
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_mean_iou",
        greater_is_better=True,
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to="none",
    )

    TrainerCls = build_iou_sft_trainer(
        SFTTrainer,
        val_samples_raw=val_dataset.samples,
        processor=processor,
        device=args.device,
        max_new_tokens=256,
    )

    trainer = TrainerCls(
        model=peft_model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    # Train
    print("\nStarting training...\n")
    trainer.train()
    print("\nTraining complete.")

    # Post-training: rename checkpoints, save best, save log
    print("\nSaving artifacts...")
    save_best_checkpoint(trainer, args.output_dir)
    rename_checkpoints_to_epoch_names(args.output_dir, args.epochs)
    save_training_log(trainer, args.output_dir)

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
