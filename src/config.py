"""Project-wide paths and hyperparameters."""

from pathlib import Path

from .dataset_profile import DATASET, IS_SNAPSHOT, SNAPSHOT_DIR

# Reproducibility
RANDOM_STATE = 42

# Train / test split
TEST_SIZE = 0.2

# Collaborative filtering
K_NEIGHBORS = 20
MIN_COMMON_RATINGS = 5
SIMILARITY_METRIC = "cosine"  # "cosine" or "pearson"

# Matrix factorization (sklearn TruncatedSVD)
SVD_N_COMPONENTS = 50
SVD_N_ITER = 5

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "ml-100k"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
if IS_SNAPSHOT:
    DATA_PROCESSED = SNAPSHOT_DIR
FIGURES_DIR = PROJECT_ROOT / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RATINGS_FILE = DATA_RAW / "u.data"
MOVIES_FILE = DATA_RAW / "u.item"
GENRE_FILE = DATA_RAW / "u.genre"

RATING_SCALE = (1, 5) if DATASET == "movielens-100k" else (0.5, 5)

# YAML configuration file (shares the same defaults as the constants above).
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path=None):
    """Load project configuration from a YAML file.

    Returns a nested dict (reproducibility/split/collaborative_filtering/svd/neural_cf/
    rating_scale). The Python constants above stay the importable defaults; this loader
    lets scripts and notebooks read the same hyperparameters from configs/config.yaml.
    """
    import yaml

    cfg_path = Path(path) if path is not None else CONFIG_FILE
    with open(cfg_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)
