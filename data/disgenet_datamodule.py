import torch
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split

class DisgenetDataModule(LightningDataModule):
    def __init__(self, batch_size = 32):
        super().__init__()
        random.seed(42)
        self.batch_size = batch_size

        self.df = None

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
        self._load_data()
        self._create_and_apply_mappings()
        self._initialize_entity_attributes()
        self._generate_negative_samples()
        self._train_test_val_split()
        

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size = self.batch_size, shuffle = True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_data, batch_size = self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size = self.batch_size)
    
    def _load_data(self) -> None:
        self.df = pd.read_csv('dga_data.csv')
        self.df['category'] = self.df['disease_id'].str[6]
        self.df['target'] = 1

    def _create_and_apply_mappings(self) -> None: 
        self.disease_id_mapping = self._create_mapping("disease_id")
        self.gene_id_mapping = self._create_mapping("gene_id")
        self.category_mapping = self._create_mapping("category")

        self.df['disease_id'] = self.df['disease_id'].map(self.disease_id_mapping)
        self.df['gene_id'] = self.df['gene_id'].map(self.gene_id_mapping)
        self.df['category'] = self.df['category'].map(self.category_mapping)

    def _initialize_entity_attributes(self) -> None:
        self.disease_attributes = [{
            "category": int(self.df[self.df['disease_id'] == i].iloc[0]["category"])
        } for i in range(len(self.disease_id_mapping))]
        self.gene_attributes = [{
            "dsi": float(self.df[self.df['gene_id'] == i].iloc[0]["dsi"]),
            "dpi": float(self.df[self.df['gene_id'] == i].iloc[0]["dpi"])
        } for i in range(len(self.gene_id_mapping))]

    def _generate_negative_samples(self) -> None: 
        adjacency_list = self.df.groupby('disease_id')['gene_id'].apply(set).to_dict()
        negative_adjacency_list = {i: set() for i in range(len(self.disease_id_mapping))}

        for _ in range(len(self.df)):
            disease_id = self._get_random_id(len(self.disease_id_mapping))
            gene_id = self._get_random_id(len(self.gene_id_mapping))
            while gene_id in adjacency_list[disease_id] or gene_id in negative_adjacency_list[disease_id]:
                gene_id = self._get_random_id(len(self.gene_id_mapping))
            negative_adjacency_list[disease_id].add(gene_id)

        for disease_id in negative_adjacency_list:
            for gene_id in negative_adjacency_list[disease_id]:
                data = {'disease_id': int(disease_id), 'gene_id': int(gene_id), "ei": 1, "dsi": self.gene_attributes[gene_id]["dsi"], "dpi": self.gene_attributes[gene_id]["dpi"], "category": int(self.disease_attributes[disease_id]["category"]), "target": int(0)}
                self.df.loc[len(self.df)] = data

    def _train_test_val_split(self) -> None:
        X = self.df.drop(columns = ['target'])  
        y = self.df['target'] 

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.3, random_state = 42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 42)

        self.train_data = self._create_hetero_data(X_train, y_train)
        self.test_data = self._create_hetero_data(X_test, y_test)
        self.val_data = self._create_hetero_data(X_val, y_val)

    def _create_mapping(self, property: str) -> Dict[Any, int]:
        return {id: index for index, id in enumerate(sorted(self.df[property].unique()))}

    def _create_hetero_data(self, X: pd.DataFrame, y: Any | pd.Series) -> HeteroData:
        data = HeteroData()
        data['disease'].id = torch.tensor([i for i in range(len(self.disease_id_mapping))])
        data["disease"].x = torch.tensor([attr["category"] for attr in self.disease_attributes], dtype = torch.float)
        data["gene"].id = torch.tensor([i for i in range(len(self.gene_id_mapping))])
        data["gene"].x = torch.tensor([[attr["dsi"], attr["dpi"]] for attr in self.gene_attributes], dtype = torch.float)
        data["disease", "to", "gene"].edge_index = torch.tensor(np.vstack((X["disease_id"].values, X["gene_id"].values)), dtype = torch.long)
        data["disease", "to", "gene"].edge_attr = torch.tensor(X["ei"].values, dtype = torch.float)
        data["disease", "to", "gene"].y = y
        return data
    
    def _get_random_id(self, range: int) -> int:
        return random.randint(0, range - 1)