import os
import torch
from model import HeteroVGAE
from data import DisgenetClient, DisgenetDataModule, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix


def prepare_data_csv():
    if not os.path.exists("dga_data.csv"):
        print("[INFO] Disgenet data not found. Preparing to create data...")
        
        disgenet_api_key = os.getenv("DISGENET_API_KEY")

        if disgenet_api_key == None:
            raise Exception("[ERROR] Missing DISGENET API key.")
    
        disgenet_client = DisgenetClient(disgenet_api_key)
        disgenet_client.create_csv_file()
        
        print("[INFO] Finished successfully.")
        
    else:
        print("[INFO] Disgenet data found. Skipping data creation.")


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

def evaluate(model, x_node, x_edge, y_truth):
    model.eval()
    output = model(x_node, x_edge)
    prediction = torch.sigmoid(output)
    
    y_true = y_truth.flatten().numpy()
    y_scores = prediction.flatten().numpy()
    
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_scores > 0.5)
    
    print(f"[EVAL] ROC AUC: {roc_auc:.6f}")
    print(f"[EVAL] PR AUC:  {pr_auc:.6f}")
    print(cm)
    
    model.train()


def main():
    prepare_data_csv()
    datamodule = DisgenetDataModule()
    datamodule.prepare_data()
    y_truth = datamodule.truth_matrix
    x_train_node, x_train_edge = initialize_data(datamodule, Dataset.TRAIN)
    x_val_node, x_val_edge = initialize_data(datamodule, Dataset.VAL)
    x_test_node, x_test_edge = initialize_data(datamodule, Dataset.TEST)
    model, optimizer, criterion = initialize_model(learning_rate = 0.001, edge_weight = datamodule.weight)

    for epoch in range(1, 200):
        optimizer.zero_grad()
        output = model(x_train_node, x_train_edge)
        loss = criterion(output, y_truth)
        print(f"[TRAIN] Epoch {epoch}: {loss.item()}")
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            with torch.no_grad():
                evaluate(model, x_val_node, x_val_edge, y_truth)
    
    with torch.no_grad():
        print("[INFO] Final test results")
        evaluate(model, x_test_node, x_test_edge, y_truth)


main()
print("[INFO] Train and test complete")
