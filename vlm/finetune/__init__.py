"""
vlmrl.vlm.finetune — LoRA finetuning pipeline for Qwen3-VL-8B-Instruct.

Modules:
    prepare_dataset  — Data preprocessing and stratified train/val splitting
    dataset          — LoRAFinetuneDataset and DETECTION_PROMPT constant
    train_lora       — LoRA training entry point
    eval_lora        — Checkpoint evaluation script
"""
