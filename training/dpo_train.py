import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

SFT_MODEL_PATH = "models/sft/pantrypal-llama-3.2-3b-sft"
OUTPUT_DIR = "models/dpo/pantrypal-llama-3.2-3b-dpo"


def load_model_and_tokenizer(model_path: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def train(dpo_data_path: str = "data/benchmark/dpo_pairs.jsonl", output_dir: str = OUTPUT_DIR) -> None:
    model, tokenizer = load_model_and_tokenizer(SFT_MODEL_PATH)
    dataset = load_dataset("json", data_files=dpo_data_path, split="train")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # when ref_model=None, trl uses the frozen base of the peft model as reference
        args=DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=1,      # DPO needs less time than SFT — 1 epoch is standard
            learning_rate=5e-5,      # lower LR than SFT — fine adjustments, not new learning
            beta=0.1,                # KL penalty weight: prevents drifting too far from SFT
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            save_strategy="epoch",
        ),
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"DPO model saved to {output_dir}")


if __name__ == "__main__":
    train()
