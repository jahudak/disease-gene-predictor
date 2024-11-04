import os

from model import Model
from data import DisgenetClient, DisgenetDataModule
from torch_geometric.data import HeteroData
import torch
from torch_geometric.nn import HeteroConv, SAGEConv, VGAE, GCNConv
from torch_geometric.nn.models.autoencoder import InnerProductDecoder


class HeteroVGAEEncoder(torch.nn.Module):
    def __init__(self, in_channels_dict, out_channels):
        super(HeteroVGAEEncoder, self).__init__()
        self.conv1 = HeteroConv(
            {
                "disease": GCNConv(1, 32, add_self_loops=False),
                "gene": GCNConv(2, 32, add_self_loops=False),
            },
            aggr="mean",
        )

        self.conv2 = HeteroConv(
            {
                "disease": GCNConv(32, 1, add_self_loops=False),
                "gene": GCNConv(32, 1, add_self_loops=False),
            },
            aggr="mean",
        )

    # def encode(self, x_dict, edge_index_dict):
    #     print("Problem starts here??")
    #     h_dict = self.encoder(x_dict, edge_index_dict)
    #     print("Problem ends here??")
    #     mu = {
    #         "disease": self.fc_mu_disease(h_dict["disease"]),
    #         "gene": self.fc_mu_gene(h_dict["gene"]),
    #     }
    #     logvar = {
    #         "disease": self.fc_logvar_disease(h_dict["disease"]),
    #         "gene": self.fc_logvar_gene(h_dict["gene"]),
    #     }
    #     return mu, logvar

    def forward(self, x_dict, edge_index_dict):
        print("Problem starts here??")
        # First layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        # Second layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


class LinkPredictionVGAE(VGAE):
    def __init__(self, in_channels_dict, out_channels):
        encoder = HeteroVGAEEncoder(in_channels_dict, out_channels)
        super().__init__(encoder)

    def decode(self, z, edge_index_dict):
        # Decode edges between disease and gene nodes
        pos_edge_index = edge_index_dict[("disease", "to", "gene")]
        z_disease, z_gene = z["disease"], z["gene"]
        pos_preds = (z_disease[pos_edge_index[0]] * z_gene[pos_edge_index[1]]).sum(
            dim=-1
        )
        return pos_preds


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
    in_channels_disease = 1  # Input features for disease nodes
    in_channels_gene = 2  # Input features for gene nodes
    out_channels = 1  # Output features for all nodes
    train_data = datamodule.train_data
    # baselineModel = HeteroVGAE(in_channels_disease, in_channels_gene, out_channels)

    disease_tensor = train_data["disease"].x
    original_shape = disease_tensor.shape[0]
    disease_tensor = disease_tensor.view(original_shape, 1)

    x_dict = {"disease": disease_tensor, "gene": train_data["gene"].x}

    edge_index_dict = {
        ("disease", "to", "gene"): train_data["disease", "to", "gene"].edge_index
    }

    # Confirm dimensions
    for node_type, features in x_dict.items():
        print(f"x_dict['{node_type}']: {features.shape}")

    # Confirm dimensions
    for edge_type, edge_index in edge_index_dict.items():
        print(f"edge_index_dict['{edge_type}']: {edge_index.shape}")

    # output = baselineModel(x_dict, edge_index_dict)
    baselineModel = LinkPredictionVGAE(1, 2)
    optimizer = torch.optim.Adam(baselineModel.parameters(), lr=0.01)


def testIterating():
    datamodule.prepare_data()
    test_dl = datamodule.test_dataloader()
    # get the dimension of node features
    # print out test_dl size

    dataloader_size = len(test_dl)
    # Print the size
    print(f"Number of batches in DataLoader: {dataloader_size}")
    print(next(iter(test_dl)))
    print(next(iter(test_dl))["disease"].x[83])
    print(next(iter(test_dl))["gene"].x[83])


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
    usingBaseLineModel()


main()
print("GYOZTUNK!!!")
# testModel()s
