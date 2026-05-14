import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from pipelines.XGB_inference_pipeline import classify_ingredients
from schemas import (
    CheckConstraintsRequest, CheckConstraintsResponse,
    ConstraintFlag, GenerateRequest, GenerateResponse,
    IngredientInferenceRequest, IngredientInferenceResponse,
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/dpo/pantrypal-llama-3.2-3b-dpo")
_model = None
_tokenizer = None


def load_model(model_path: str = MODEL_PATH):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb, device_map="auto"
    )
    return model, tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer
    _model, _tokenizer = load_model()
    yield


app = FastAPI(title="PantryPal", lifespan=lifespan)


def _generate(instruction: str) -> str:
    import torch
    messages = [{"role": "user", "content": instruction}]
    input_ids = _tokenizer.apply_chat_template(messages, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        output = _model.generate(
            input_ids, max_new_tokens=512, do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    return _tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)


def _xgb_check(text: str) -> list[ConstraintFlag]:
    result = classify_ingredients([text[:500]])
    return [
        ConstraintFlag(flag=k, probability=v["probability"], present=v["present"])
        for k, v in result.items()
    ]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    restriction_str = ", ".join(req.restrictions) if req.restrictions else "no specific restrictions"
    instruction = (
        f"Generate a recipe that is {restriction_str}. "
        f"Use these ingredients: {', '.join(req.pantry_items)}"
    )
    if req.cuisine:
        instruction += f" Make it {req.cuisine} cuisine."

    recipe = _generate(instruction)
    flags = _xgb_check(recipe)

    # Single retry if XGBoost detects a stated restriction was violated
    stated = set(r.replace("-free", "").replace("low-", "high_") for r in req.restrictions)
    if any(f.present and f.flag in stated for f in flags):
        recipe = _generate(instruction)
        flags = _xgb_check(recipe)

    return GenerateResponse(recipe=recipe, constraint_check=flags)


@app.post("/check_constraints", response_model=CheckConstraintsResponse)
def check_constraints(req: CheckConstraintsRequest):
    result = classify_ingredients(req.ingredients)
    return CheckConstraintsResponse(flags=[
        ConstraintFlag(flag=k, probability=v["probability"], present=v["present"])
        for k, v in result.items()
    ])


# Original XGBoost inference endpoint — preserved from v1
@app.post("/ingredient_inference", response_model=IngredientInferenceResponse)
async def ingredient_inference(request: IngredientInferenceRequest):
    if not request.ingredients:
        raise HTTPException(status_code=400, detail="Ingredients list must be nonempty")
    if len(request.ingredients) > 100:
        raise HTTPException(status_code=413, detail="Request payload too large; ingredient list exceeds 100 items.")
    return {"labels": classify_ingredients(request.ingredients)}
