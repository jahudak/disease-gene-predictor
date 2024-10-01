import os
from model import Model
from data import DisgenetClient

def main():
    disgenet_api_key = os.getenv("DISGENET_API_KEY")
    example_value = os.getenv("EXAMPLE_KEY")

    if(disgenet_api_key == None):
        print("[ERROR] Missing DISGENET API key.")
        return

    disgenet_client = DisgenetClient(disgenet_api_key)
    model = Model(disgenet_client)
    model.test_imports()
    model.test_data()

    print(example_value)

main()