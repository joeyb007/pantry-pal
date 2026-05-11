import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

MAX_SEQ_LEN = 1024
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "models/sft/pantrypal-llama-3.2-3b-sft"


def load_model_and_tokenizer(model_name: str = BASE_MODEL):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def format_prompt(example: dict, tokenizer) -> dict:
    # Apply the model's native chat template so training tokens match inference tokens exactly.
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def train(data_dir: str = "data/benchmark", output_dir: str = OUTPUT_DIR) -> None:
    model, tokenizer = load_model_and_tokenizer()

    dataset = load_dataset("json", data_files={
        "train": f"{data_dir}/train.jsonl",
        "validation": f"{data_dir}/val.jsonl",
    })
    dataset = dataset.map(lambda ex: format_prompt(ex, tokenizer))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,    # effective batch = 16
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            max_seq_length=MAX_SEQ_LEN,
            dataset_text_field="text",
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"SFT model saved to {output_dir}")


if __name__ == "__main__":
    train()
