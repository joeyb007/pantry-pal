import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOTrainer, DPOConfig

# Patches trl's DPOTrainer to use unsloth's memory-efficient implementation
PatchDPOTrainer()

SFT_MODEL_PATH = "models/sft/pantrypal-llama-3.2-3b-sft"
OUTPUT_DIR = "models/dpo/pantrypal-llama-3.2-3b-dpo"


def train(dpo_data_path: str = "data/benchmark/dpo_pairs.jsonl", output_dir: str = OUTPUT_DIR) -> None:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SFT_MODEL_PATH,
        max_seq_length=1024,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
    )

    dataset = load_dataset("json", data_files=dpo_data_path, split="train")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # unsloth derives reference model from the SFT checkpoint internally
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
