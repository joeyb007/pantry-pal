import pandas as pd
from pathlib import Path
# Goal: manipulate cleaned recipes into JSON data for model training.
BASE_DIR = Path(__file__).resolve().parents[2]
PATH_TO_SEED_LABELS = BASE_DIR / 'data' / 'CSV_data' / 'seed_labeling_unlabeled.csv'
PATH_TO_CLEANED_DATA = BASE_DIR / 'data' / 'CSV_data' / 'recipes_cleaned.csv'

# Removing seed labels from cleaned data
seed_labeled_data_df = pd.read_csv(PATH_TO_SEED_LABELS)
original_indices = seed_labeled_data_df['original_index']
full_df = pd.read_csv(PATH_TO_CLEANED_DATA)
filtered_df = full_df[~full_df.index.isin(original_indices)]
filtered_df = filtered_df.reset_index(drop=True)

# Next, run models on this dataframe and create new rows
from models.XGB_models.load_XGB import load_models

XGB_models = load_models()

if __name__ == '__main__':
    # Should be exactly 10,000 less rows (removing all seed labels)
    print(type(filtered_df))