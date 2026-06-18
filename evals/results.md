# PantryPal Eval Results

## Constraint Satisfaction Rate (CSR)

| Model | CSR | Violations |
|-------|-----|------------|
| Base Llama 3.2 3B | 54.92% | 435/965 |
| PantryPal SFT | 99.69% | 3/965 |
| PantryPal DPO | **72.02%** | 270/965 |
| GPT-4o | 56.79% | 417/965 |

## Recipe Quality (GPT-4o-as-Judge, n=50)

| Model | Score (1-10) |
|-------|-------------|
| PantryPal SFT | 6.96 |
| PantryPal DPO | 6.68 |
| GPT-4o | 7.92 |

## Training Delta

SFT vs base: +44.77pp (54.92% → 99.69%)
DPO vs SFT:  -27.67pp (99.69% → 72.02%)

## Per-Restriction CSR Breakdown (SFT model)

| Restriction | CSR |
|-------------|-----|
| alcohol | 100.0% |
| beef | 100.0% |
| chicken | 100.0% |
| dairy | 91.3% |
| eggs | 100.0% |
| fish | 100.0% |
| gluten | 100.0% |
| high_carb | 95.45% |
| honey | 100.0% |
| legumes | 100.0% |
| nuts | 100.0% |
| peanuts | 100.0% |
| pork | 100.0% |
| processed_meats | 100.0% |
| sesame | 100.0% |
| shellfish | 100.0% |
| soy | 100.0% |
| sugar | 100.0% |
