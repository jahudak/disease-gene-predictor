from enum import Enum

# Simple enum class to represent the dataset type
class Dataset(Enum):
    TRAIN = "train_data"
    TEST = "test_data"
    VAL = "val_data"