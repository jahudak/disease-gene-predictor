import os

from model import HeteroVGAE
from data import DisgenetClient, DisgenetDataModule
import torch


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
    in_channels_disease = 2  # Input features for disease nodes
    in_channels_gene = 2  # Input features for gene nodes
    out_channels = 1  # Output features for all nodes
    train_data = datamodule.train_data
    baselineModel = HeteroVGAE(in_channels_disease, in_channels_gene, out_channels)

    # dataModuleDf = datamodule.df
    # geneAttributes = dataModuleDf[["gene_id", "gene_name", "gene_description"]]
    # geneAttributes["gene_id"] = dataModuleDf["gene_id"]
    # create an ID for each gene
    # geneAttributes["gene_idx"] = geneAttributes.to_numpy()
    # geneAttributes["gene_dpi"] = geneAttributes["dpi"]
    # geneAttributes["gene_dsi"] = geneAttributes["dsi"]

    disease_categories = train_data["disease"].x.view(-1, 1)

    # Create an ID tensor incrementing from 0 to n
    num_nodes = train_data["disease"].x.shape[0]
    disease_tensor = torch.arange(num_nodes).view(-1, 1)

    # print("Disease tensor shape:")
    # print(disease_tensor.shape)
    # print("-----------------")
    # print(disease_tensor)
    # print("-----------------")
    # print("Disease categories shape:")
    # print(disease_categories.shape)
    # print("-----------------")
    # print(disease_categories)
    # print("-----------------")

    concatanated_tensor = torch.cat((disease_tensor, disease_categories), dim=1)

    # print("Concatanated tensor shape:")
    # print(concatanated_tensor.shape)
    # print("-----------------")
    # print(concatanated_tensor)
    # print("-----------------")

    x_dict = {"disease": concatanated_tensor, "gene": train_data["gene"].x}

    edge_index_dict = {
        ("disease", "to", "gene"): train_data["disease", "to", "gene"].edge_index
    }

    # Confirm dimensions
    for node_type, features in x_dict.items():
        print(f"x_dict['{node_type}']: {features.shape}")

    # Confirm dimensions
    for edge_type, edge_index in edge_index_dict.items():
        print(f"edge_index_dict['{edge_type}']: {edge_index.shape}")

    # print("-----------------")
    # print(x_dict)
    # print("-----------------")

    output = baselineModel(x_dict, edge_index_dict)
    # Train the model
    baselineModel.train()

    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(baselineModel.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Iterate over the training data
    for epoch in range(10):
        optimizer.zero_grad()
        output = baselineModel(x_dict, edge_index_dict)
        loss = criterion(output, train_data["disease"].y)
        loss.backward()
        optimizer.step()

    # Evaluate the model
    baselineModel.eval()
    with torch.no_grad():
        output = baselineModel(x_dict, edge_index_dict)
        predictions = torch.sigmoid(output)
        accuracy = torch.mean((predictions > 0.5).float() == train_data["disease"].y)
        print(f"Accuracy: {accuracy.item()}")


def testIterating():
    datamodule.prepare_data()
    test_dl = datamodule.test_dataloader()
    # get the dimension of node features
    # print out test_dl size

    dataloader_size = len(test_dl)
    # Print the size
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
    # testModel()
    # testDataModule()
    # dataLoaderIterationTest()
    # testIterating()
    datamodule.prepare_data()
    usingBaseLineModel()


main()
print("GYOZTUNK!!!")
# testModel()s
