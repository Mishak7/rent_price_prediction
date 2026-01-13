import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGET = "price"

FEATURES = [
    'total_area',
    'renovation',
    'parking',
    'lat',
    'lon',
    'building_type',
    'room_type',
    'rarity_index',
    'flag_big_area',
    'rooms_count',
    'loggia_count',
    'floor_ratio',
    'city',
    'exp',
    'dist_center',
    'log_area',
    'area_sq',
    'lux_anchor_flag'
]

CAT_FEATURES = [
    'room_type',
    'building_type',
    'parking',
    'renovation',
    'city'
]

SEEDS = [11, 22, 33, 44, 55]

