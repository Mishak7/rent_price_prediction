import pandas as pd
import joblib
import os

from train_and_test.features_engineering import prepare_data


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "server", "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def build_artifacts():
    print("Loading train data...")
    train_df = pd.read_excel(os.path.join(BASE_DIR, "data/raw/train.xlsx"))

    print("Preparing train data...")
    train_df = prepare_data(train_df)

    train_df['total_area'] = pd.to_numeric(train_df['total_area'], errors='coerce')
    train_df['rooms_count'] = pd.to_numeric(train_df['rooms_count'], errors='coerce')

    train_df['total_area'] = train_df['total_area'].fillna(train_df['total_area'].median())
    train_df['rooms_count'] = train_df['rooms_count'].fillna(train_df['rooms_count'].median())

    train_df['area_room_combo'] = (
            train_df['total_area']
            .round()
            .astype(int)
            .astype(str)
            + '_'
            + train_df['rooms_count']
            .round()
            .astype(int)
            .astype(str)
    )

    combo_freq = train_df['area_room_combo'].value_counts().to_dict()
    joblib.dump(combo_freq, os.path.join(ARTIFACTS_DIR, "combo_freq.pkl"))

    area_threshold = train_df['total_area'].quantile(0.9)
    joblib.dump(area_threshold, os.path.join(ARTIFACTS_DIR, "area_threshold.pkl"))

    if 'dist_center' not in train_df.columns:
        raise ValueError("dist_center not found in train_df — check prepare_data")

    dist_center_median = train_df['dist_center'].median()
    joblib.dump(dist_center_median, os.path.join(ARTIFACTS_DIR, "dist_center_median.pkl"))


if __name__ == "__main__":
    build_artifacts()
