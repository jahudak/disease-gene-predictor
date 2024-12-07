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
            raise Exception("[ERROR] Missing DISGENET API key.")
    
        disgenet_client = DisgenetClient(disgenet_api_key)
        disgenet_client.create_csv_file()
        
        print("[LOG] Finished successfully.")
        
    else:
        print("[LOG] Disgenet data found. Skipping data creation.")


def main():
    prepare_data_csv()
    datamodule = DisgenetDataModule()
    datamodule.prepare_data()
    
    train_data = datamodule.train_data
    
    x_dict = {
        "disease": train_data["disease"].x, 
        "gene": train_data["gene"].x
    }
    x_dict["disease"] = x_dict["disease"].view(-1, 1)

    edge_index_dict = {
        ("disease", "to", "gene"): train_data["disease", "to", "gene"].edge_index,
        ("gene", "rev_to", "disease"): train_data["gene", "rev_to", "disease"].edge_index,
    }
    edge_index_adj = datamodule.truth_matrix

    baselineModel = HeteroVGAE(in_channels_disease = 1, in_channels_gene = 2, out_channels = 1)
    optimizer = torch.optim.Adam(baselineModel.parameters(), lr = 0.001)
    criterion = torch.nn.BCEWithLogitsLoss(weight = datamodule.weight)

    for epoch in range(1, 200 + 1):
        optimizer.zero_grad()
        output = baselineModel(x_dict, edge_index_dict)
        loss = criterion(output, edge_index_adj)
        print(f"[{epoch}]: {loss.item()}")
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            with torch.no_grad():
                pred = torch.sigmoid(output)

                y_true = edge_index_adj.flatten().numpy()
                y_scores = pred.flatten().numpy()
                
                auc = roc_auc_score(y_true, y_scores)
                pr_auc = average_precision_score(y_true, y_scores)
                cm = confusion_matrix(y_true, y_scores > 0.5)
                
                print(f"ROC AUC: {auc}")
                print(f"PR AUC:  {pr_auc}")
                print(cm)


main()
print("[LOG] Train and test complete")
