from fastapi import FastAPI
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import os
import logging
from pydantic import BaseModel

from server.FlatRawInfo import FlatRawInfo
from server.adapter import adapter
from train_and_test.postproccesing import calibrate_predictions
from train_and_test.features_engineering import prepare_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class FlatPrediction(BaseModel):
    listing_id: Optional[int]
    predicted_price: Optional[float]

FEATURES = ['total_area', 'renovation', 'parking', 'lat', 'lon', 'building_type', 'room_type', 'rarity_index', 'flag_big_area', 'rooms_count', 'loggia_count', 'floor_ratio', 'city', 'exp', 'dist_center', 'log_area', 'area_sq', 'lux_anchor_flag' ]

logger.info("Loading training data...")
train_df = pd.read_excel(os.path.join(BASE_DIR, "data/raw/train.xlsx"))
logger.info("Preparing training data...")
train_df = prepare_data(train_df)

@app.post("/predict", response_model=List[FlatPrediction])
def predict(flat: FlatRawInfo):
    listing_id = flat.listing_id

    df = pd.DataFrame([flat.model_dump()])
    df_prepared = adapter(df)

    model_path = os.path.join(BASE_DIR, "model", "catboost_model.pkl")
    models = joblib.load(model_path)

    prediction = df_prepared[FEATURES]
    raw_prediction = np.zeros(len(prediction))
    for model in models:
        raw_prediction += np.expm1(model.predict(prediction)) / len(models)
    logger.info(f"raw: {raw_prediction}")

    final_prediction = calibrate_predictions(raw_prediction,
                                             train_df,
                                             df_prepared)

    prediction = FlatPrediction(listing_id=listing_id, predicted_price=float(final_prediction[0]))

    return [prediction]

