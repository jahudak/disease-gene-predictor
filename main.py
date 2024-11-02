import os

from model import Model
from data import DisgenetClient, DisgenetDataModule


def dgaData():
    disgenet_api_key = os.getenv("DISGENET_API_KEY")
    # example_value = os.getenv("EXAMPLE_KEY")
    # example_value = os.getenv("EXAMPLE_KEY")

    # if disgenet_api_key == None:
    if disgenet_api_key == None:
        print("[ERROR] Missing DISGENET API key.")
        return

    if not os.path.exists("dga_data.csv"):
        print("[LOG] Disgenet data not found. Preparing to create data...")
        disgenet_client = DisgenetClient(disgenet_api_key)
        disgenet_client.create_csv_file()

    datamodule = DisgenetDataModule()
    print("[LOG] Finished successfully")


def testModel():
    disgenet_client = DisgenetClient(os.getenv("DISGENET_API_KEY"))
    modelInstance = Model(disgenet_client)
    modelInstance.test_imports()
    # modelInstance.test_data()


def main():
    global disgenet_client, datamodule
    if not os.path.exists("dga_data.csv"):
        dgaData()
    else:
        print("[LOG] Disgenet data found. Skipping data creation.")
    testModel()


main()
# testModel()
