import os

from model import HeteroVGAE
from data import DisgenetClient, DisgenetDataModule
import torch


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
    in_channels_disease = 2     # Input features for disease nodes (id and category)
    in_channels_gene = 2        # Input features for gene nodes (dsi, dpi)
    out_channels = 1            # Output features for all nodes (is there a link?)
    global train_data
    train_data = datamodule.train_data
    global baselineModel
    baselineModel = HeteroVGAE(in_channels_disease, in_channels_gene, out_channels)

    disease_categories = train_data["disease"].x.view(-1, 1)

    # Adding the disease tensor category value (integer) to the disease tensor
    num_nodes = train_data["disease"].x.shape[0]
    disease_tensor = torch.arange(num_nodes).view(-1, 1)

    concatanated_tensor = torch.cat((disease_tensor, disease_categories), dim=1)

    global x_dict
    x_dict = {"disease": concatanated_tensor, "gene": train_data["gene"].x}

    global edge_index_dict
    edge_index_dict = {
        ("disease", "to", "gene"): train_data["disease", "to", "gene"].edge_index
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
    global disgenet_client, datamodule
    datamodule = DisgenetDataModule()
    if not os.path.exists("dga_data.csv"):
        dgaData()
    else:
        print("[LOG] Disgenet data found. Skipping data creation.")
    
    datamodule.prepare_data()
    prepareTransformation()
    usingBaseLineModel()


main()
print("[LOG] Train and test complete")
