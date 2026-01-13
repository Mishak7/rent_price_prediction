import pandas as pd
import numpy as np
import joblib
import logging
import os

from train_and_test.features_engineering import prepare_data
from train_and_test.postproccesing import calibrate_predictions
from train_and_test.config import FEATURES, BASE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict():
    logger.info("Loading test data...")
    test_df = pd.read_excel(os.path.join(BASE_DIR, "data/raw/test.xlsx"))
    logger.info("Preparing test data...")
    test_df = prepare_data(test_df)

    logger.info("Loading train data for calibration...")
    train_df = pd.read_excel(os.path.join(BASE_DIR, "data/raw/train.xlsx"))
    train_df = prepare_data(train_df)

    logger.info("Loading trained models...")
    models = joblib.load(os.path.join(BASE_DIR, "model/catboost_model.pkl"))

    X_test = test_df[FEATURES]

    logger.info("Generating raw predictions...")
    raw_preds = np.zeros(len(X_test))
    for model in models:
        raw_preds += np.expm1(model.predict(X_test)) / len(models)

    logger.info("Calibrating predictions...")
    final_preds = calibrate_predictions(
        preds=raw_preds,
        df_train=train_df,
        df_test=test_df
    )

    test_df['predicted_price'] = final_preds
    logger.info("Predictions ready.")

    return test_df


if __name__ == "__main__":
    df_result = predict()
    df_result.to_csv(os.path.join(BASE_DIR, "data/processed/predictions.csv"), index=False)
    logger.info("Predictions saved to data/processed/predictions.csv")
