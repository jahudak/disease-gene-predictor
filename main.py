import os

from model import Model
from data import DisgenetClient, DisgenetDataModule
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningModule, Trainer
from torch_geometric.data import HeteroData


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, 2 * out_channels)

    def forward(self, x, edge_index):
        # print(f"Input x shape: {x.shape}")
        x = F.relu(self.conv1(x, edge_index))
        # print(f"Shape after conv1: {x.shape}")
        x = self.conv2(x, edge_index)
        # print(f"Shape after conv2: {x.shape}")
        return x.chunk(2, dim=-1)


class VGAEModel(LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VGAEModel, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.vgae = VGAE(self.encoder)

    def forward(self, data: HeteroData):
        x, edge_index = data["disease"].x, data["disease", "to", "gene"].edge_index
        print(f"x shape: {x.shape}")
        print(f"edge_index shape: {edge_index.shape}")
        return self.vgae.encode(x, edge_index)

    def training_step(self, batch, batch_idx):
        z = self(batch)
        loss = self.vgae.recon_loss(z, batch["disease", "to", "gene"].edge_index)
        loss = loss + (1 / batch["disease"].x.size(0)) * self.vgae.kl_loss()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        z = self(batch)
        loss = self.vgae.recon_loss(z, batch["disease", "to", "gene"].edge_index)
        loss = loss + (1 / batch["disease"].x.size(0)) * self.vgae.kl_loss()
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        z = self(batch)
        loss = self.vgae.recon_loss(z, batch["disease", "to", "gene"].edge_index)
        loss = loss + (1 / batch["disease"].x.size(0)) * self.vgae.kl_loss()
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


def dgaData():
    disgenet_api_key = os.getenv("DISGENET_API_KEY")
    # example_value = os.getenv("EXAMPLE_KEY")

    if disgenet_api_key == None:
        print("[ERROR] Missing DISGENET API key.")
        return

    if not os.path.exists("dga_data.csv"):
        print("[LOG] Disgenet data not found. Preparing to create data...")
        disgenet_client = DisgenetClient(disgenet_api_key)
        disgenet_client.create_csv_file()
    print("[LOG] Finished successfully")


def usingBaseLineModel():
    dimension_dl = datamodule.test_dataloader()
    node_features = 1
    print(f"Node features: {node_features}")
    hidden_dim = 16
    latent_space_dim = 8
    baseline_model = VGAEModel(
        input_dim=node_features, hidden_dim=hidden_dim, output_dim=latent_space_dim
    )
    trainer = Trainer(max_epochs=10)
    print("TRAINING ELKEZDODOTT!!!\n")
    trainer.fit(
        baseline_model, datamodule.train_dataloader(), datamodule.val_dataloader()
    )
    trainer.test(baseline_model, datamodule.test_dataloader())


def testIterating():
    datamodule.prepare_data()
    test_dl = datamodule.test_dataloader()
    # get the dimension of node features
    print("Dimension of node features \n")
    print(next(iter(test_dl))["disease"].x.shape)
    print(next(iter(test_dl)))
    print(next(iter(test_dl)))  # this gives an error


def main():
    global disgenet_client, datamodule
    datamodule = DisgenetDataModule()
    if not os.path.exists("dga_data.csv"):
        dgaData()
    else:
        print("[LOG] Disgenet data found. Skipping data creation.")
    # testModel()
    # testDataModule()
    # dataLoaderIterationTest()
    testIterating()
    # datamodule.prepare_data()
    # usingBaseLineModel()


main()
# testModel()s
