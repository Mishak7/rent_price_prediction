import pandas as pd
import numpy as np
import joblib
import logging
import os

from catboost import CatBoostRegressor
from train_and_test.features_engineering import prepare_data
from train_and_test.config import FEATURES, CAT_FEATURES, TARGET, SEEDS, BASE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train():
    logger.info("Loading training data...")
    train_df = pd.read_excel(os.path.join(BASE_DIR, "data/raw/train.xlsx"))
    logger.info("Preparing training data...")
    train_df = prepare_data(train_df)

    X = train_df[FEATURES]
    y = np.log1p(train_df[TARGET])

    models = []
    for seed in SEEDS:
        logger.info(f"Training model with seed {seed}...")
        model = CatBoostRegressor(
            iterations=1900,
            learning_rate=0.05,
            depth=7,
            eval_metric='MAE',
            random_seed=seed,
            verbose=False
        )
        model.fit(X, y, cat_features=CAT_FEATURES)
        models.append(model)

    logger.info("Saving models to model/catboost_model.pkl...")
    joblib.dump(models, os.path.join(BASE_DIR, "model/catboost_model.pkl"))
    logger.info("Training finished.")


if __name__ == "__main__":
    train()


