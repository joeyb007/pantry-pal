import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MAX_SEQ_LEN = 1024
BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "models/sft/pantrypal-llama-3.2-3b-sft"


def load_model_and_tokenizer(model_name: str = BASE_MODEL):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # required for gradient checkpointing + PEFT
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
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
        processing_class=tokenizer,
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
