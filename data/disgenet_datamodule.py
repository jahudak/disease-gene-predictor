import pandas as pd
import torch
import random
from torch_geometric.data import HeteroData
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

class DisgenetDataModule(LightningDataModule):
    def __init__(self, batch_size = 32):
        super().__init__()
        random.seed(42)
        self.batch_size = batch_size
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.prepare_data()

    def prepare_data(self):
        df = pd.read_csv('dga_data.csv')
        df['category'] = df['disease_id'].str[6]
        df['target'] = 1

        disease_id_mapping = {id_str: idx for idx, id_str in enumerate(df['disease_id'].unique())}
        gene_id_mapping = {id_str: idx for idx, id_str in enumerate(sorted(df['gene_id'].unique()))}
        category_mapping = {id_str: idx for idx, id_str in enumerate(sorted(df['category'].unique()))}

        df['disease_id'] = df['disease_id'].map(disease_id_mapping)
        df['gene_id'] = df['gene_id'].map(gene_id_mapping)
        df['category'] = df['category'].map(category_mapping)

        disease_attributes = [{
            "category": int(df[df['disease_id'] == i].iloc[0]["category"])
        } for i in range(len(disease_id_mapping))]
        gene_attributes = [{
            "dsi": float(df[df['gene_id'] == i].iloc[0]["dsi"]),
            "dpi": float(df[df['gene_id'] == i].iloc[0]["dpi"])
        } for i in range(len(gene_id_mapping))]

        adjacency_list = df.groupby('disease_id')['gene_id'].apply(set).to_dict()
        negative_adjacency_list = {i: set() for i in range(len(disease_id_mapping))}

        for _ in range(len(df)):
            disease_id = random.randint(0, len(disease_id_mapping) - 1)
            gene_id = random.randint(0, len(gene_id_mapping) - 1)
            while gene_id in adjacency_list[disease_id] or gene_id in negative_adjacency_list[disease_id]:
                gene_id = random.randint(0, len(gene_id_mapping) - 1)
            negative_adjacency_list[disease_id].add(gene_id)

        for disease_id in negative_adjacency_list:
            for gene_id in negative_adjacency_list[disease_id]:
                data = {'disease_id': int(disease_id), 'gene_id': int(gene_id), "ei": 1, "dsi": gene_attributes[gene_id]["dsi"], "dpi": gene_attributes[gene_id]["dpi"], "category": int(disease_attributes[disease_id]["category"]), "target": int(0)}
                df.loc[len(df)] = data

        X = df.drop(columns = ['target'])  
        y = df['target'] 

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = 0.3, random_state = 42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 42)

        train_data = HeteroData()
        train_data['disease'].id = torch.tensor([i for i in range(len(disease_id_mapping))])
        train_data["disease"].x = torch.tensor([attr["category"] for attr in disease_attributes], dtype = torch.float)
        train_data["gene"].id = torch.tensor([i for i in range(len(gene_id_mapping))])
        train_data["gene"].x = torch.tensor([[attr["dsi"], attr["dpi"]] for attr in gene_attributes], dtype = torch.float)
        train_data["disease", "to", "gene"].edge_index = torch.tensor([X_train["disease_id"].values, X_train["gene_id"].values], dtype = torch.long)
        train_data["disease", "to", "gene"].edge_attr = torch.tensor(X_train["ei"].values, dtype = torch.float)
        train_data["disease", "to", "gene"].y = y_train
        self.train_data = train_data

        test_data = HeteroData()
        test_data['disease'].id = torch.tensor([i for i in range(len(disease_id_mapping))])
        test_data["disease"].x = torch.tensor([attr["category"] for attr in disease_attributes], dtype = torch.float)
        test_data["gene"].id = torch.tensor([i for i in range(len(gene_id_mapping))])
        test_data["gene"].x = torch.tensor([[attr["dsi"], attr["dpi"]] for attr in gene_attributes], dtype = torch.float)
        test_data["disease", "to", "gene"].edge_index = torch.tensor([X_test["disease_id"].values, X_test["gene_id"].values], dtype = torch.long)
        test_data["disease", "to", "gene"].edge_attr = torch.tensor(X_test["ei"].values, dtype = torch.float)
        test_data["disease", "to", "gene"].y = y_test
        self.test_data = test_data

        val_data = HeteroData()
        val_data['disease'].id = torch.tensor([i for i in range(len(disease_id_mapping))])
        val_data["disease"].x = torch.tensor([attr["category"] for attr in disease_attributes], dtype = torch.float)
        val_data["gene"].id = torch.tensor([i for i in range(len(gene_id_mapping))])
        val_data["gene"].x = torch.tensor([[attr["dsi"], attr["dpi"]] for attr in gene_attributes], dtype = torch.float)
        val_data["disease", "to", "gene"].edge_index = torch.tensor([X_val["disease_id"].values, X_val["gene_id"].values], dtype = torch.long)
        val_data["disease", "to", "gene"].edge_attr = torch.tensor(X_val["ei"].values, dtype = torch.float)
        val_data["disease", "to", "gene"].y = y_val
        self.val_data = val_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size = self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size = self.batch_size)