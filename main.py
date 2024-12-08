import os
import torch
from model import HeteroVGAE
from data import DisgenetClient, DisgenetDataModule, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix


def prepare_data_csv():
    if not os.path.exists("dga_data.csv"):
        print("[LOG] Disgenet data not found. Preparing to create data...")
        
        disgenet_api_key = os.getenv("DISGENET_API_KEY")

        if disgenet_api_key == None:
            raise Exception("[ERROR] Missing DISGENET API key.")
    
        disgenet_client = DisgenetClient(disgenet_api_key)
        disgenet_client.create_csv_file()
        
        print("[LOG] Finished successfully.")
        
    else:
        print("[LOG] Disgenet data found. Skipping data creation.")


def initialize_data(datamodule: DisgenetDataModule, dataset: Dataset):
    data = getattr(datamodule, dataset.value, None)
    if data is None:
        raise ValueError(f"Invalid dataset_type '{dataset}'.")
    
    node_features = {
        "disease": data["disease"].x, 
        "gene": data["gene"].x
    }
    node_features["disease"] = node_features["disease"].view(-1, 1)

    edge_features = {
        ("disease", "to", "gene"): data["disease", "to", "gene"].edge_index,
        ("gene", "rev_to", "disease"): data["gene", "rev_to", "disease"].edge_index,
    }
    
    return node_features, edge_features


def initialize_model(learning_rate, edge_weight):
    model = HeteroVGAE(in_channels_disease = 1, in_channels_gene = 2, out_channels = 1)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss(weight = edge_weight)
    
    return model, optimizer, criterion


def main():
    prepare_data_csv()
    datamodule = DisgenetDataModule()
    datamodule.prepare_data()
    y_truth = datamodule.truth_matrix
    x_train, y_train = initialize_data(datamodule, Dataset.TRAIN)
    model, optimizer, criterion = initialize_model(learning_rate = 0.001, edge_weight = datamodule.weight)

    for epoch in range(1, 200 + 1):
        optimizer.zero_grad()
        output = model(x_train, y_train)
        loss = criterion(output, y_truth)
        print(f"[{epoch}]: {loss.item()}")
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            with torch.no_grad():
                pred = torch.sigmoid(output)

                y_true = y_truth.flatten().numpy()
                y_scores = pred.flatten().numpy()
                
                auc = roc_auc_score(y_true, y_scores)
                pr_auc = average_precision_score(y_true, y_scores)
                cm = confusion_matrix(y_true, y_scores > 0.5)
                
                print(f"ROC AUC: {auc}")
                print(f"PR AUC:  {pr_auc}")
                print(cm)


main()
print("[LOG] Train and test complete")
