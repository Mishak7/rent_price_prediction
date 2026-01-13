import pandas as pd
import numpy as np
import joblib
import os


ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__),"artifacts")

COMBO_FREQ = joblib.load(os.path.join(ARTIFACTS_DIR, "combo_freq.pkl"))
AREA_THRESHOLD = joblib.load(os.path.join(ARTIFACTS_DIR, "area_threshold.pkl"))
DIST_CENTER_MEDIAN = joblib.load(os.path.join(ARTIFACTS_DIR, "dist_center_median.pkl"))


def adapter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    categorical_cols = ['room_type', 'building_type', 'parking', 'renovation', 'city']
    for col in categorical_cols:
        df[col] = df[col].fillna('unknown').astype(str)

    numeric_cols = ['total_area', 'rooms_count', 'lat', 'lon', 'floor', 'floors_total']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['total_area'] = df['total_area'].fillna(60)
    df['rooms_count'] = df['rooms_count'].fillna(2)
    df['lat'] = df['lat'].fillna(np.nan)
    df['lon'] = df['lon'].fillna(np.nan)
    df['floor'] = df['floor'].fillna(5)
    df['floors_total'] = df['floors_total'].replace(0, np.nan).fillna(10)

    df['log_area'] = np.log1p(df['total_area'])
    df['area_sq'] = df['total_area'] ** 2
    df['flag_big_area'] = (df['total_area'] > AREA_THRESHOLD).astype(int)

    df['floor_ratio'] = df['floor'] / df['floors_total']
    df['floor_ratio'] = df['floor_ratio'].clip(0, 1)

    df['area_room_combo'] = (
        df['total_area'].round().astype(int).astype(str) + '_' +
        df['rooms_count'].astype(int).astype(str)
    )

    df['combo_frequency'] = df['area_room_combo'].map(COMBO_FREQ).fillna(0)
    df['rarity_index'] = 1 / (df['combo_frequency'] + 1)

    lux_anchor_keywords = [
        'двухуровнев', 'двухэтаж', 'панорам',
        'француз', 'терраса', 'консьерж',
        'valet', 'лобби', 'residence',
    ]

    df['lux_anchor_flag'] = (
        df.get('description', '')
        .fillna('')
        .str.lower()
        .apply(lambda x: int(any(k in x for k in lux_anchor_keywords)))
    )

    is_moscow = (df['city'] == 'Москва').astype(int)
    good_renovation = df['renovation'].isin(['Дизайнерский', 'Евроремонт']).astype(int)

    df['exp'] = (
        df['total_area']
        * is_moscow
        * good_renovation
        * df['rarity_index']
    )

    CITY_CENTERS = {
        'Москва': (55.7558, 37.6173),
        'Санкт-Петербург': (59.9343, 30.3351),
        'Свердловская область': (56.8389, 60.6057),
    }

    def geo_dist(row):
        if row['city'] not in CITY_CENTERS:
            return np.nan
        lat0, lon0 = CITY_CENTERS[row['city']]
        if pd.isna(row['lat']) or pd.isna(row['lon']):
            return np.nan
        return np.sqrt((row['lat'] - lat0) ** 2 + (row['lon'] - lon0) ** 2)

    df['dist_center'] = df.apply(geo_dist, axis=1)
    df['dist_center'] = df['dist_center'].fillna(DIST_CENTER_MEDIAN)

    return df
