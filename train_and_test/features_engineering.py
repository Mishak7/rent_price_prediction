import pandas as pd
import numpy as np

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_and_basic_parse(df)
    df = address_features(df)
    df = rooms(df)
    df = floors(df)
    df = balcony_features(df)
    df = extras_features(df)
    df = area_features(df)
    df = elevator(df)
    df = bathroom(df)
    df = complex_features(df)
    df = rarity_index(df)
    df = lux_anchor(df)
    df = cleanup(df)
    df = expensive_extra(df)
    df = geo_features(df)
    return df


def rename_and_basic_parse(df: pd.DataFrame) -> pd.DataFrame:
    rename_dict = {
        'ID  объявления': 'listing_id',
        'Количество комнат': 'rooms_count',
        'Тип': 'property_type',
        'Метро': 'metro',
        'Адрес': 'address',
        'Площадь, м2': 'area_m2',
        'Дом': 'building',
        'Парковка': 'parking',
        'Описание': 'description',
        'Ремонт': 'renovation',
        'Площадь комнат, м2': 'rooms_area',
        'Балкон': 'balcony',
        'Окна': 'windows',
        'Санузел': 'bathroom',
        'Можно с детьми/животными': 'kids_pets_allowed',
        'Дополнительно': 'extras',
        'Название ЖК': 'complex_name',
        'Серия дома': 'building_series',
        'Высота потолков, м': 'ceiling_height',
        'Лифт': 'elevator',
        'Мусоропровод': 'garbage_chute',
        'Price': 'price',
        'Комнаты': 'rooms',
        'Площадь_общая': 'total_area',
        'Этаж': 'floor',
        'Этажность': 'floors_total',
        'Метро_станция': 'metro_station',
        'Метро_минуты': 'metro_minutes',
        'Метро_тип': 'metro_transport'
    }
    return df.rename(columns=rename_dict)


def address_features(df: pd.DataFrame) -> pd.DataFrame:
    df['city'] = df['address'].str.split(', ').str[0]
    street_keywords = [
        'улица', 'набережная', 'проспект',
        'бульвар', 'шоссе', 'переулок',
    ]
    df['street'] = df['address'].str.split(', ').apply(
        lambda parts: next(
            (p for p in parts if any(k in p for k in street_keywords)),
            None,
        )
    )
    return df


def rooms(df: pd.DataFrame) -> pd.DataFrame:
    df['room_type'] = df['rooms_count'].str.split(', ').str[1]
    room_numbers = df['rooms_count'].str.split(', ').str[0]
    df['rooms_count'] = pd.to_numeric(room_numbers, errors='coerce')
    return df


def balcony_features(df: pd.DataFrame) -> pd.DataFrame:
    def parse_balcony(s):
        if pd.isna(s):
            return pd.Series([0, 0, 0, 0])
        has_balcony = balcony_count = 0
        has_loggia = loggia_count = 0
        for part in s.split(', '):
            if 'Балкон' in part:
                has_balcony = 1
                num = ''.join(filter(str.isdigit, part))
                balcony_count = int(num) if num else 0
            elif 'Лоджия' in part:
                has_loggia = 1
                num = ''.join(filter(str.isdigit, part))
                loggia_count = int(num) if num else 0
        return pd.Series([has_balcony, balcony_count, has_loggia, loggia_count])

    df[
        ['has_balcony', 'balcony_count', 'has_loggia', 'loggia_count']
    ] = df['balcony'].apply(parse_balcony)
    return df


def extras_features(df: pd.DataFrame) -> pd.DataFrame:
    extras_dict = {
        'Ванна': 'bath',
        'Душевая кабина': 'shower',
        'Интернет': 'internet',
        'Кондиционер': 'air_conditioner',
        'Мебель в комнатах': 'furniture_rooms',
        'Мебель на кухне': 'furniture_kitchen',
        'Посудомоечная машина': 'dishwasher',
        'Стиральная машина': 'washing_machine',
        'Телевизор': 'tv',
        'Телефон': 'phone',
        'Холодильник': 'fridge',
    }
    for ru, en in extras_dict.items():
        df[en] = df['extras'].str.contains(ru, na=False).astype(int)
    return df


def area_features(df: pd.DataFrame) -> pd.DataFrame:
    df['log_area'] = np.log1p(df['total_area'])
    df['area_sq'] = df['total_area'] ** 2
    area_threshold = df['total_area'].quantile(0.9)
    df['flag_big_area'] = (df['total_area'] > area_threshold).astype(int)
    return df


def elevator(df: pd.DataFrame) -> pd.DataFrame:
    def parse_elevator(s):
        if pd.isna(s):
            return pd.Series([0, 0, 0, 0])
        has_pass = pass_count = 0
        has_freight = freight_count = 0
        for part in s.split(', '):
            if 'Пасс' in part:
                has_pass = 1
                num = ''.join(filter(str.isdigit, part))
                pass_count = int(num) if num else 0
            elif 'Груз' in part:
                has_freight = 1
                num = ''.join(filter(str.isdigit, part))
                freight_count = int(num) if num else 0
        return pd.Series([has_pass, pass_count, has_freight, freight_count])

    df[
        [
            'has_passenger_elevator',
            'passenger_elevator_count',
            'has_freight_elevator',
            'freight_elevator_count',
        ]
    ] = df['elevator'].apply(parse_elevator)
    return df


def bathroom(df: pd.DataFrame) -> pd.DataFrame:
    def parse_bathroom(s):
        if pd.isna(s):
            return pd.Series([0, 0, 0, 0])
        has_combined = combined_count = 0
        has_separate = separate_count = 0
        for part in s.split(', '):
            if 'Совмещенный' in part:
                has_combined = 1
                num = ''.join(filter(str.isdigit, part))
                combined_count = int(num) if num else 0
            elif 'Раздельный' in part:
                has_separate = 1
                num = ''.join(filter(str.isdigit, part))
                separate_count = int(num) if num else 0
        return pd.Series([has_combined, combined_count, has_separate, separate_count])

    df[
        [
            'has_combined_bathroom',
            'combined_bathroom_count',
            'has_separate_bathroom',
            'separate_bathroom_count',
        ]
    ] = df['bathroom'].apply(parse_bathroom)
    return df


def complex_features(df: pd.DataFrame) -> pd.DataFrame:
    df['complex'] = df['complex_name'].str.split(', ').str[0]
    df['building_type'] = df['building'].str.split(', ').str[1]
    return df


def floors(df: pd.DataFrame) -> pd.DataFrame:
    df['floor_ratio'] = df['floor'] / df['floors_total']
    return df


def rarity_index(df: pd.DataFrame) -> pd.DataFrame:
    df['area_room_combo'] = df['total_area'].astype(str).str[:3] + '_' + df['rooms'].astype(str)
    combo_counts = df['area_room_combo'].value_counts()
    df['combo_frequency'] = df['area_room_combo'].map(combo_counts)
    df['rarity_index'] = 1 / (df['combo_frequency'] + 1)
    return df


def lux_anchor(df: pd.DataFrame) -> pd.DataFrame:
    lux_anchor_keywords = [
        'двухуровнев', 'двухэтаж', 'панорамное остекление',
        'французские окна', 'терраса на крыше', 'консьерж',
        'valet', 'лобби', 'residence',
    ]
    df['lux_anchor_flag'] = (
        df['description']
        .str.lower()
        .fillna('')
        .apply(lambda x: any(k in x for k in lux_anchor_keywords))
        .astype(int)
    )
    return df


def cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(
        columns=[
            'property_type', 'metro', 'address', 'area_m2', 'building', 'rooms_area',
            'balcony', 'windows', 'bathroom', 'kids_pets_allowed', 'extras',
            'building_series', 'elevator', 'garbage_chute', 'metro_station', 'metro_transport',
            'floor', 'floors_total', 'area_room_combo', 'phone', 'internet', 'complex_name', 'description'
        ]
    )
    categorical_cols = ['room_type', 'building_type', 'parking', 'renovation', 'complex']
    for col in categorical_cols:
        df[col] = df[col].fillna('unknown')
    numeric_cols = ['ceiling_height', 'rooms', 'lat', 'lon', 'metro_minutes']
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    return df


def expensive_extra(df: pd.DataFrame) -> pd.DataFrame:
    df['exp'] = df['total_area'] * (df['city'] == 'Москва') * (
        (df['renovation'] == 'Дизайнерский') | (df['renovation'] == 'Евроремонт')
    ) * df['rarity_index']
    return df


def geo_features(df: pd.DataFrame) -> pd.DataFrame:
    CITY_CENTERS = {
        'Москва': (55.7558, 37.6173),
        'Санкт-Петербург': (59.9343, 30.3351),
        'Свердловская область': (56.8389, 60.6057),
    }

    def geo_dist(row):
        if row['city'] not in CITY_CENTERS:
            return np.nan
        lat0, lon0 = CITY_CENTERS[row['city']]
        return np.sqrt((row['lat'] - lat0)**2 + (row['lon'] - lon0)**2)

    df['dist_center'] = df.apply(geo_dist, axis=1)
    return df
