import os

from model import HeteroVGAE
from data import DisgenetClient, DisgenetDataModule
import torch
from torch_geometric.nn import HeteroConv, SAGEConv, VGAE
from torch_geometric.nn.models.autoencoder import InnerProductDecoder
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def dgaData():
    disgenet_api_key = os.getenv("DISGENET_API_KEY")

    if disgenet_api_key == None:
        print("[ERROR] Missing DISGENET API key.")
        return

    if not os.path.exists("dga_data.csv"):
        print("[LOG] Disgenet data not found. Preparing to create data...")
        disgenet_client = DisgenetClient(disgenet_api_key)
        disgenet_client.create_csv_file()
    print("[LOG] Finished successfully")


def prepareTransformation():
    in_channels_disease = 1  # Input features for disease nodes (id and category)
    in_channels_gene = 2  # Input features for gene nodes (dsi, dpi)
    out_channels = 1  # Output features for all nodes (is there a link?)
    global train_data
    train_data = datamodule.train_data
    global baselineModel
    baselineModel = HeteroVGAE(in_channels_disease, in_channels_gene, out_channels)

    # disease_categories = train_data["disease"].x.view(-1, 1)

    # Adding the disease tensor category value (integer) to the disease tensor
    # num_nodes = train_data["disease"].x.shape[0]
    # disease_tensor = torch.arange(num_nodes).view(-1, 1)

    # concatanated_tensor = torch.cat((disease_tensor, disease_categories), dim=1)

    global x_dict
    x_dict = {"disease": train_data["disease"].x, "gene": train_data["gene"].x}

    global edge_index_dict
    edge_index_dict = {
        ("disease", "to", "gene"): train_data["disease", "to", "gene"].edge_index,
        ("gene", "rev_to", "disease"): train_data[
            "gene", "rev_to", "disease"
        ].edge_index,
    }


def usingBaseLineModel():
    output = baselineModel(x_dict, edge_index_dict)
    baselineModel.train(x_dict, edge_index_dict)
    optimizer = torch.optim.Adam(baselineModel.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(10):
        optimizer.zero_grad()
        output = baselineModel(x_dict, edge_index_dict)
        loss = criterion(output, train_data["disease"].x)
        loss.backward()
        optimizer.step()

    baselineModel.eval()
    with torch.no_grad():
        output = baselineModel(x_dict, edge_index_dict)
        pred = torch.sigmoid(output)
        accuracy = torch.mean((pred > 0.5).float() == train_data["disease"].y)
        print(f"Accuracy: {accuracy.item()}")


def testIterating():
    datamodule.prepare_data()
    test_dl = datamodule.test_dataloader()

    dataloader_size = len(test_dl)
    print(f"Number of batches in DataLoader: {dataloader_size}")

    print(next(iter(test_dl))["disease"].x[83])
    print(next(iter(test_dl))["gene"].x[83])


def main():
    global disgenet_client, datamodule, x_dict, edge_index_dict
    datamodule = DisgenetDataModule()
    if not os.path.exists("dga_data.csv"):
        dgaData()
    else:
        print("[LOG] Disgenet data found. Skipping data creation.")

    datamodule.prepare_data()
    prepareTransformation()
    # # usingBaseLineModel()
    # torch.save(x_dict, "x_dict.pt")
    # torch.save(edge_index_dict, "edge_index_dict.pt")
    # x_dict = torch.load("x_dict.pt")
    # edge_index_dict = torch.load("edge_index_dict.pt")

    # print(x_dict["disease"].shape)
    # print(x_dict["gene"].shape)
    # print(edge_index_dict.keys())
    # print(edge_index_dict["disease", "to", "gene"].shape)
    # print(edge_index_dict["gene", "rev_to", "disease"].shape)

    x_dict["disease"] = x_dict["disease"].view(-1, 1)
    """
    print(edge_index_dict["disease", "to", "gene"])
    print(edge_index_dict["gene", "rev_to", "disease"])

    edge_index_adj = to_dense_adj(edge_index_dict["disease", "to", "gene"])
    edge_index_adj_rev = to_dense_adj(edge_index_dict["gene", "rev_to", "disease"])

    print("************")
    print(edge_index_adj.sum())
    print(edge_index_adj.shape)
    print(edge_index_dict["disease", "to", "gene"].shape)
    print(edge_index_adj_rev.sum())
    print(edge_index_adj_rev.shape)
    print(edge_index_dict["gene", "rev_to", "disease"].shape)
    print("************")
    """
    edge_index_adj = datamodule.get_valami()
    print("************")
    print(edge_index_adj.sum())
    print(edge_index_adj.shape)
    print("************")

    # encoder = HeteroConv(
    #     {
    #         ("disease", "to", "gene"): SAGEConv(
    #             [1, 2],
    #             10,
    #             # FYI: It doesnt like when the data is indexed from 0
    #         ),
    #         ("gene", "rev_to", "disease"): SAGEConv([2, 1], 10),
    #     },
    #     aggr="sum",
    # )
    # x_dict1 = encoder(x_dict, edge_index_dict)
    baselineModel = HeteroVGAE(1, 2, 1)
    output = baselineModel(x_dict, edge_index_dict)
    optimizer = torch.optim.Adam(baselineModel.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss(weight=datamodule.get_weight())

    for epoch in range(200):
        optimizer.zero_grad()
        output = baselineModel(x_dict, edge_index_dict)
        # print(output.shape)
        loss = criterion(output, edge_index_adj)
        print("************")
        print(epoch)
        print(loss.item())
        print("************")

        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            with torch.no_grad():
                pred = torch.sigmoid(output)

                y_true = edge_index_adj.flatten().numpy()
                y_scores = pred.flatten().numpy()
                print(y_true)
                auc = roc_auc_score(y_true, y_scores)
                print(f"ROC AUC: {auc}")
                pr_auc = average_precision_score(y_true, y_scores)
                print(f"PR AUC: {pr_auc}")
                cm = confusion_matrix(y_true, y_scores > 0.5)
                print(cm)


# accuracy = torch.mean(((pred > 0.5).float() == edge_index_adj).float())
# print(f"Accuracy: {accuracy.item()}")

# print(x_dict1["disease"].shape)
# print(x_dict1["gene"].shape)


main()
print("[LOG] Train and test complete")
