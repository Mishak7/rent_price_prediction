import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_street_median_shrink(df_train: pd.DataFrame,
                             df_test: pd.DataFrame) -> pd.Series:
    logger.info("Calculating street median shrink...")
    moscow_street_median = (
        df_train[df_train['city'] == 'Москва']
        .groupby('street')['price']
        .median()
    )
    street_median = df_test['street'].map(moscow_street_median)
    global_median = df_train['price'].median()
    return street_median.fillna(global_median)


def blend_expensive_flats(predictions: np.ndarray,
                          df_test: pd.DataFrame,
                          street_median: pd.Series,
                          alpha: float = 0.25) -> np.ndarray:
    threshold = np.percentile(predictions, 90)
    high_price_mask = (df_test['city'] == 'Москва') & (predictions > threshold)
    logger.info(f"Blending {high_price_mask.sum()} expensive flats...")
    blended = predictions.copy()
    blended[high_price_mask] = (
        (1 - alpha) * predictions[high_price_mask] +
        alpha * street_median.loc[high_price_mask]
    )
    return blended


def apply_complex_corrections(predictions: np.ndarray,
                              df_test: pd.DataFrame) -> np.ndarray:
    complex_medians = {
        "Четыре солнца": 3_700_000,
        "Парк Палас": 2_300_000,
        "Прайм Парк": 850_000,
        "Смоленская Застава": 850_000,
        "Дом на Озерковской": 650_000,
        "Новопесковский": 550_000,
        "Меркурий Тауэр": 575_000,
        "Четыре Ветра": 490_000,
        "Триумф-Палас": 450_000,
        "Созвездие Капитал-1": 420_000,
        "Клубный дом Печатников": 395_000,
        "Поклонная 9": 330_000,
        "Созвездие Капитал-2": 270_000
    }
    factor = 0.9
    median_increase = 1.05
    corrected = predictions.copy()
    for complex_name, median_price in complex_medians.items():
        mask = df_test['complex'] == complex_name
        low_price_mask = mask & (corrected < median_price * factor)
        corrected[low_price_mask] = median_price * median_increase
        logger.info(f"Corrected {low_price_mask.sum()} flats in {complex_name}")
    return corrected


def round_prices(predictions: np.ndarray, step: int = 5000) -> np.ndarray:
    return np.floor(predictions / step) * step


def calibrate_predictions(predictions: np.ndarray,
                          df_train: pd.DataFrame,
                          df_test: pd.DataFrame) -> np.ndarray:
    logger.info("Starting prediction calibration...")
    street_median = add_street_median_shrink(df_train, df_test)
    predictions = blend_expensive_flats(predictions, df_test, street_median)
    predictions = apply_complex_corrections(predictions, df_test)
    predictions = round_prices(predictions)
    logger.info("Calibration done.")
    return predictions
