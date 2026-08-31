#!/usr/bin/env python3
"""
TRL-based SFT for the RS-CRAFT 7B run (rebuttal).

Stack: HuggingFace TRL SFTTrainer + accelerate-launch + DeepSpeed ZeRO-2.

This is the *proven* SFT recipe used for the paper's SFT-CRAFT models
(originally at CRAFT_package/src/sft.py). We use it here to bypass the
ms-swift 4.1.3 + DeepSpeed regression that produced loss=0 from step 2.

Launch via:
    accelerate launch \\
        --config_file cfg/accelerate_deepspeed_zero2.yaml \\
        scripts/train/sft_trl.py \\
        --model_name_or_path Qwen/Qwen2.5-7B-Instruct \\
        --dataset_name data/train/rscraft/rscraft_best_of_4_v1.jsonl \\
        ... (see scripts/train/sft_trl.sh)
"""
from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


def main():
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    quantization_config = get_quantization_config(model_config)
    # Pass torch_dtype as a string so TRL's JSON arg-logging doesn't choke.
    dtype_str = str(model_config.torch_dtype).replace("torch.", "") if model_config.torch_dtype else "bfloat16"
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        dtype=dtype_str,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=script_args.dataset_name)
    train_size = len(dataset["train"])
    if train_size < 5000:
        dataset = dataset["train"].train_test_split(test_size=0.1)
    else:
        dataset = dataset["train"].train_test_split(test_size=500)
    print(f"train dataset size: {len(dataset['train'])}")
    print(f"test dataset size : {len(dataset['test'])}")

    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    main()
