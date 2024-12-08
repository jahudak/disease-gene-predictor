import os
import torch
from model import HeteroVGAE
from data import DisgenetClient, DisgenetDataModule, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
import optuna
from optuna_dashboard import run_server
import datetime


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

    node_features = {"disease": data["disease"].x, "gene": data["gene"].x}
    node_features["disease"] = node_features["disease"].view(-1, 1)

    edge_features = {
        ("disease", "to", "gene"): data["disease", "to", "gene"].edge_index,
        ("gene", "rev_to", "disease"): data["gene", "rev_to", "disease"].edge_index,
    }

    return node_features, edge_features


def objective(trial, datamodule: DisgenetDataModule):
    y_truth = datamodule.truth_matrix
    x_train_node, x_train_edge = initialize_data(datamodule, Dataset.TRAIN)
    x_val_node, x_val_edge = initialize_data(datamodule, Dataset.VAL)
    x_test_node, x_test_edge = initialize_data(datamodule, Dataset.TEST)
    edge_weight = datamodule.weight

    learning_rate = trial.suggest_float("learning", 1e-5, 1e-1, log=False)
    heterovgae_hidden_channels = trial.suggest_int("heterovgae_hidden_channels", 8, 32)
    encoder_hidden_channels = trial.suggest_int("encoder_hidden_channels", 8, 32)
    optimizier = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    model, optimizer, criterion = initialize_model(
        learning_rate,
        heterovgae_hidden_channels,
        encoder_hidden_channels,
        optimizier,
        edge_weight,
    )

    best_accuracy = 0.0

    for epoch in range(1, 200):
        optimizer.zero_grad()
        output = model(x_train_node, x_train_edge)
        loss = criterion(output, y_truth)
        print(f"[TRAIN] Epoch {epoch}: {loss.item()}")
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                accuracy = evaluate(model, x_val_node, x_val_edge, y_truth)
                best_accuracy = max(best_accuracy, accuracy)

    return best_accuracy


def optimize_hyperparameters(datamodule: DisgenetDataModule):
    db_storage = "sqlite:///db.sqlite3"
    current_datetime = datetime.datetime.now()
    study_name = f"disgenet_study_{current_datetime.strftime('%Y-%m-%d_%H-%M-%S')}"
    study = optuna.create_study(
        direction="maximize",
        storage=db_storage,  # Specify the storage URL here.
        study_name=study_name,
    )
    study.optimize(lambda trial: objective(trial, datamodule), n_trials=50, n_jobs=-1)
    run_server(db_storage)
    print(f"Best parameters are: {study.best_params}")
    return study.best_params


def initialize_model(
    learning_rate,
    heterovgae_hidden_channels_param,
    encoder_hidden_channels_param,
    optimizier,
    edge_weight,
):
    model = HeteroVGAE(
        in_channels_disease=1,
        in_channels_gene=2,
        out_channels=1,
        heterovgae_hidden_channels=heterovgae_hidden_channels_param,  # can be optimized
        encoder_hidden_channels=encoder_hidden_channels_param,  # can be optimized
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss(weight=edge_weight)

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

    # Accuracy calculation based on the correct prediction
    y_pred = (y_scores > 0.5).astype(int)
    accuracy = (y_pred == y_true).mean()
    print(f"[EVAL] Accuracy: {accuracy:.6f}")
    model.train()
    return accuracy


def main():
    prepare_data_csv()
    datamodule = DisgenetDataModule()
    datamodule.prepare_data()

    optimize_hyperparameters(datamodule)

    # y_truth = datamodule.truth_matrix
    # x_train_node, x_train_edge = initialize_data(datamodule, Dataset.TRAIN)
    # x_val_node, x_val_edge = initialize_data(datamodule, Dataset.VAL)
    # x_test_node, x_test_edge = initialize_data(datamodule, Dataset.TEST)
    # model, optimizer, criterion = initialize_model(
    #     learning_rate=0.001, edge_weight=datamodule.weight
    # )

    # for epoch in range(1, 200):
    #     optimizer.zero_grad()
    #     output = model(x_train_node, x_train_edge)
    #     loss = criterion(output, y_truth)
    #     print(f"[TRAIN] Epoch {epoch}: {loss.item()}")
    #     loss.backward()
    #     optimizer.step()

    #     if epoch % 10 == 0:
    #         with torch.no_grad():
    #             evaluate(model, x_val_node, x_val_edge, y_truth)

    # with torch.no_grad():
    #     print("[INFO] Final test results")
    #     evaluate(model, x_test_node, x_test_edge, y_truth)


main()
print("[INFO] Train and test complete")
