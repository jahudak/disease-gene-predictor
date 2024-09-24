import torch
import lightning
import torch_geometric as ptg
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from data import DisgenetClient

class Model():
    def __init__(self):
        self.disgenet_client = DisgenetClient("mocked_api_key")
    
    def test_imports(self):
        print(f"PyTorch version: {torch.__version__}")
        print(f"Lightning version: {lightning.__version__}")
        print(f"PyTorch Geometric version: {ptg.__version__}")
        print(f"Numpy version: {np.__version__}")
        print(f"Pandas version: {pd.__version__}")
        print(f"Matplotlib version: {matplotlib.__version__}")
        print(f"Seaborn version: {sns.__version__}")

    def test_data(self):
        print(self.disgenet_client.get_data())
