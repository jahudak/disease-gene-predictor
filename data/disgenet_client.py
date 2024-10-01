class DisgenetClient: 
    def __init__(self, api_key: str):
        self.api_key = api_key 

    def get_data(self):
        return f"{self.api_key}:mocked_data"