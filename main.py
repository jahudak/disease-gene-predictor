import os
import torch
import gradio as gr
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


def initialize_model(learning_rate, weight_decay, edge_weight, encoder_hidden_channels, encoder_out_channels):
    model = HeteroVGAE(
        in_channels_disease = 1, 
        in_channels_gene = 2, 
        encoder_hidden_channels = encoder_hidden_channels, 
        encoder_out_channels = encoder_out_channels, 
        out_channels = 1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
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
    
    return roc_auc, pr_auc


def train_and_evaluate_model(learning_rate, weight_decay, encoder_hidden_channels, encoder_out_channels):
    prepare_data_csv()
    datamodule = DisgenetDataModule()
    datamodule.prepare_data()
    y_truth = datamodule.truth_matrix
    x_train_node, x_train_edge = initialize_data(datamodule, Dataset.TRAIN)
    x_val_node, x_val_edge = initialize_data(datamodule, Dataset.VAL)
    x_test_node, x_test_edge = initialize_data(datamodule, Dataset.TEST)
    model, optimizer, criterion = initialize_model(learning_rate, weight_decay, datamodule.weight, encoder_hidden_channels, encoder_out_channels)

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
        return evaluate(model, x_test_node, x_test_edge, y_truth)


def main():
    def run_model(learning_rate, weight_decay, encoder_hidden_channels, encoder_out_channels):
        return train_and_evaluate_model(learning_rate, weight_decay, encoder_hidden_channels, encoder_out_channels)
    
    demo = gr.Interface(
        fn=run_model,
        inputs=[
            gr.Slider(
                label="Learning rate", 
                value=0.001, 
                minimum=0.001, 
                maximum=0.1, 
                step=0.001
            ),
            gr.Slider(
                label="Weight decay", 
                value=0.0, 
                minimum=0.0, 
                maximum=0.001, 
                step=0.0001
            ),
            gr.Slider(
                label="Encoder hidden channels", 
                value=64, 
                minimum=4, 
                maximum=128, 
                step=1
            ),
            gr.Slider(
                label="Encoder out channels", 
                value=32, 
                minimum=2, 
                maximum=64, 
                step=1
            ),
        ],
        outputs=[
            gr.Textbox(
                label="ROC AUC"
            ), 
            gr.Textbox(
                label="PR AUC"
            )
        ]
    )

    demo.launch()


main()