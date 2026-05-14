# Outlines JSON type schemas for requests and responses
from pydantic import BaseModel
from typing import List, Dict

class IngredientInferenceRequest(BaseModel):
    ingredients: List[str]

class IngredientLabel(BaseModel):
    probability: float
    present: bool

class IngredientInferenceResponse(BaseModel):
    labels: Dict[str, IngredientLabel]

class LLMRequest(BaseModel):
    ingredients_on_hand: List[str]
    forbidden_ingredients: Dict[str, int]
    preferences: str

class LLMResponse(BaseModel):
    title: str
    ingredients: List[str]
    directions: List[str]

# New schemas for the /generate and /check_constraints endpoints
from typing import Optional

class GenerateRequest(BaseModel):
    restrictions: List[str]
    pantry_items: List[str]
    cuisine: Optional[str] = None

class ConstraintFlag(BaseModel):
    flag: str
    probability: float
    present: bool

class GenerateResponse(BaseModel):
    recipe: str
    constraint_check: List[ConstraintFlag]

class CheckConstraintsRequest(BaseModel):
    ingredients: List[str]

class CheckConstraintsResponse(BaseModel):
    flags: List[ConstraintFlag]
    