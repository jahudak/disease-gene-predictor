import os
import torch
from model import HeteroVGAE
from data import DisgenetClient, DisgenetDataModule
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix


def prepare_data_csv():
    if not os.path.exists("dga_data.csv"):
        print("[LOG] Disgenet data not found. Preparing to create data...")
        
        disgenet_api_key = os.getenv("DISGENET_API_KEY")

        if disgenet_api_key == None:
            print("[ERROR] Missing DISGENET API key.")
            return
    
        disgenet_client = DisgenetClient(disgenet_api_key)
        disgenet_client.create_csv_file()
        
        print("[LOG] Finished successfully")
        
    else:
        print("[LOG] Disgenet data found. Skipping data creation.")


def prepareTransformation():
    global train_data
    train_data = datamodule.train_data
    global baselineModel
    baselineModel = HeteroVGAE(in_channels_disease = 1, in_channels_gene = 2, out_channels = 1)

    global x_dict
    x_dict = {"disease": train_data["disease"].x, "gene": train_data["gene"].x}

    global edge_index_dict
    edge_index_dict = {
        ("disease", "to", "gene"): train_data["disease", "to", "gene"].edge_index,
        ("gene", "rev_to", "disease"): train_data["gene", "rev_to", "disease"].edge_index,
    }


def usingBaseLineModel():
    output = baselineModel(x_dict, edge_index_dict)
    baselineModel.train(x_dict, edge_index_dict)
    optimizer = torch.optim.Adam(baselineModel.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    for _ in range(10):
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


def main():
    global disgenet_client, datamodule, x_dict, edge_index_dict
    datamodule = DisgenetDataModule()
    prepare_data_csv()
    datamodule.prepare_data()
    prepareTransformation()

    x_dict["disease"] = x_dict["disease"].view(-1, 1)
    
    edge_index_adj = datamodule.get_truth_matrix()
    print("************")
    print(edge_index_adj.sum())
    print(edge_index_adj.shape)
    print("************")

    baselineModel = HeteroVGAE(1, 2, 1)
    output = baselineModel(x_dict, edge_index_dict)
    optimizer = torch.optim.Adam(baselineModel.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss(weight=datamodule.get_weight())

    for epoch in range(200):
        optimizer.zero_grad()
        output = baselineModel(x_dict, edge_index_dict)
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


main()
print("[LOG] Train and test complete")
