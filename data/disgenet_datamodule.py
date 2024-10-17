import torch
import random
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

class DisgenetDataModule(LightningDataModule):
    def __init__(self, batch_size = 32):
        super().__init__()
        random.seed(42)
        self.batch_size = batch_size
        self.train_data = None
        self.test_data = None
        self.val_data = None

        self.disease_id_mapping = None
        self.gene_id_mapping = None
        self.category_mapping = None

        self.disease_attributes = None
        self.gene_attributes = None

        self.prepare_data()

    def prepare_data(self):
        df = pd.read_csv('dga_data.csv')
        df['category'] = df['disease_id'].str[6]
        df['target'] = 1

        self.disease_id_mapping = {id_str: idx for idx, id_str in enumerate(df['disease_id'].unique())}
        self.gene_id_mapping = {id_str: idx for idx, id_str in enumerate(sorted(df['gene_id'].unique()))}
        self.category_mapping = {id_str: idx for idx, id_str in enumerate(sorted(df['category'].unique()))}

        df['disease_id'] = df['disease_id'].map(self.disease_id_mapping)
        df['gene_id'] = df['gene_id'].map(self.gene_id_mapping)
        df['category'] = df['category'].map(self.category_mapping)

        self.disease_attributes = [{
            "category": int(df[df['disease_id'] == i].iloc[0]["category"])
        } for i in range(len(self.disease_id_mapping))]
        self.gene_attributes = [{
            "dsi": float(df[df['gene_id'] == i].iloc[0]["dsi"]),
            "dpi": float(df[df['gene_id'] == i].iloc[0]["dpi"])
        } for i in range(len(self.gene_id_mapping))]

        adjacency_list = df.groupby('disease_id')['gene_id'].apply(set).to_dict()
        negative_adjacency_list = {i: set() for i in range(len(self.disease_id_mapping))}

        for _ in range(len(df)):
            disease_id = random.randint(0, len(self.disease_id_mapping) - 1)
            gene_id = random.randint(0, len(self.gene_id_mapping) - 1)
            while gene_id in adjacency_list[disease_id] or gene_id in negative_adjacency_list[disease_id]:
                gene_id = random.randint(0, len(self.gene_id_mapping) - 1)
            negative_adjacency_list[disease_id].add(gene_id)

        for disease_id in negative_adjacency_list:
            for gene_id in negative_adjacency_list[disease_id]:
                data = {'disease_id': int(disease_id), 'gene_id': int(gene_id), "ei": 1, "dsi": self.gene_attributes[gene_id]["dsi"], "dpi": self.gene_attributes[gene_id]["dpi"], "category": int(self.disease_attributes[disease_id]["category"]), "target": int(0)}
                df.loc[len(df)] = data

        X = df.drop(columns = ['target'])  
        y = df['target'] 

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.3, random_state = 42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 42)

        self.train_data = self._create_hetero_data(X_train, y_train)
        self.test_data = self._create_hetero_data(X_test, y_test)
        self.val_data = self._create_hetero_data(X_val, y_val)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size = self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size = self.batch_size)

    def _create_hetero_data(self, X, y) -> HeteroData:
        data = HeteroData()
        data['disease'].id = torch.tensor([i for i in range(len(self.disease_id_mapping))])
        data["disease"].x = torch.tensor([attr["category"] for attr in self.disease_attributes], dtype = torch.float)
        data["gene"].id = torch.tensor([i for i in range(len(self.gene_id_mapping))])
        data["gene"].x = torch.tensor([[attr["dsi"], attr["dpi"]] for attr in self.gene_attributes], dtype = torch.float)
        data["disease", "to", "gene"].edge_index = torch.tensor(np.vstack((X["disease_id"].values, X["gene_id"].values)), dtype = torch.long)
        data["disease", "to", "gene"].edge_attr = torch.tensor(X["ei"].values, dtype = torch.float)
        data["disease", "to", "gene"].y = y
        return data