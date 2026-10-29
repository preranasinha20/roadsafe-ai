"""Shared paths and constants for the road-safety project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "Accident_Information.csv"
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
TAB_DIR = ROOT / "reports" / "tables"
APP_DATA = ROOT / "app" / "public" / "data"

SEED = 42

# Time-based split: train on the past, validate on the next year, test on the latest years.
TRAIN_YEARS = range(2005, 2015)   # 2005-2014
VAL_YEARS = [2015]
TEST_YEARS = [2016, 2017]

SEVERITY_ORDER = ["Slight", "Serious", "Fatal"]        # 0, 1, 2
SEV_MAP = {name: i for i, name in enumerate(SEVERITY_ORDER)}
# EPDO-style weights used for black spots (Fatal 10, Serious 5, Slight 1)
SEV_WEIGHT = {"Slight": 1, "Serious": 5, "Fatal": 10}

UK_BBOX = dict(lat=(49.8, 60.9), lon=(-8.7, 1.8))

GRID_DEG = 0.005   # about 550 m in latitude, grid used for historical risk features

for d in (PROC_DIR, MODELS_DIR, FIG_DIR, TAB_DIR, APP_DATA):
    d.mkdir(parents=True, exist_ok=True)
