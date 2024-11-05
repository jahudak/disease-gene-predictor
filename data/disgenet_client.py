import requests
import json
import time
import math
from typing import Dict, List, Any


class DisgenetClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_calls = 0
        self.total_results = 0
        self.disease_count = 0
        self.gene_count = 0
        self.disease_file_path = "disease_ids.txt"
        self.data_file_path = "dga_data.csv"
        self.disgenet_base_url = "https://api.disgenet.com"
        self.disgenet_dga_route = "/api/v1/gda/summary"

    def create_csv_file(self) -> None:
        print("[LOG] Starting data creation process. This may take a few minutes...")

        start_time = time.time()
        self.write_to_csv_file(["disease_id,gene_id,ei,dsi,dpi"])

        disease_categories: List[str] = self.get_disease_categories()
        for disease_category in disease_categories:
            results: List[str] = []
            params: Dict = {"disease": self.get_disease_param(disease_category)}

            response = self.send_request(params, self.disgenet_dga_route)
            self.process_response(response, results)
            pages_left = math.ceil(response["paging"]["totalElements"] / 100) - 1

            for page in range(1, pages_left + 1):
                params["page_number"] = page
                response = self.send_request(params, self.disgenet_dga_route)
                self.process_response(response, results)

        self.postprocess_csv_file()
        end_time = time.time()
        self.csv_log(start_time, end_time)
        return

    def send_request(self, params: Dict, route: str) -> Any:
        self.api_calls += 1
        headers: Dict = {"Authorization": self.api_key, "accept": "application/json"}
        response = requests.get(
            url=self.disgenet_base_url + route, params=params, headers=headers
        )
        response = self.handle_api_rate_limit(response, params, route)
        return json.loads(response.text)

    def process_response(self, response: Dict, results: List[str]) -> None:
        for association in response["payload"]:
            disease_id: str = ""
            for id in association["diseaseVocabularies"]:
                if id.startswith("ICD10_") and len(id) == 9:
                    disease_id = id
                    break

            ei = round(association["ei"], 3) if association["ei"] is not None else 0
            dsi = (
                association["geneDSI"] if association["geneDSI"] is not None else 0.230
            )
            dpi = association["geneDPI"] if association["geneDPI"] is not None else 0

            results.append(
                f"{disease_id},{association["symbolOfGene"]},{ei},{dsi},{dpi}"
            )
        self.write_to_csv_file(results)
        results.clear()
        return

    def handle_api_rate_limit(self, response: Dict, params: Dict, route: str) -> Any:
        if response.status_code == 429:
            while response.status_code == 429:
                wait_time = (
                    int(response.headers["x-rate-limit-retry-after-seconds"]) + 1
                )
                print(f"[LOG] Waiting {wait_time} seconds to restore rate limit")
                time.sleep(wait_time)

                self.api_calls += 1
                headers: Dict = {
                    "Authorization": self.api_key,
                    "accept": "application/json",
                }
                response = requests.get(
                    url=self.disgenet_base_url + route, params=params, headers=headers
                )

                if response.ok:
                    break
                else:
                    continue
        return response

    def write_to_csv_file(self, data: List[str]) -> None:
        with open(self.data_file_path, "a") as file:
            for line in data:
                file.write(line + "\n")
        return

    def postprocess_csv_file(self) -> None:
        with open(self.data_file_path, "r") as file:
            lines = file.readlines()

        header: str = lines[0]
        rows: List[str] = lines[1:]

        sorted_rows: List[str] = sorted(
            rows, key=lambda line: line.strip().split(",")[0]
        )
        self.total_results = len(sorted_rows)

        self.disease_count = len({line.strip().split(",")[0] for line in sorted_rows})
        self.gene_count = len({line.strip().split(",")[1] for line in sorted_rows})

        with open(self.data_file_path, "w") as file:
            file.write(header)
            file.writelines(sorted_rows)
        return

    def get_disease_param(self, disease_ids: str) -> str:
        category: str = disease_ids[0]
        disease_range: str = disease_ids[1:]
        start: int = int(disease_range.split("-")[0])
        end: int = int(disease_range.split("-")[1])

        disease_param: str = ""
        for i in range(start, end + 1):
            index: str = str(i) if i >= 10 else "0" + str(i)
            disease_param += "ICD10_" + category + index
            if i != end:
                disease_param += ","
        return disease_param

    def get_disease_categories(self) -> List[str]:
        with open(self.disease_file_path, "r") as file:
            categories = file.readlines()
        return categories

    def csv_log(self, start_time, end_time) -> None:
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        print(
            f"[LOG] Data file created with {self.disease_count} diseases, {self.gene_count} genes and {self.total_results} associations."
        )
        if minutes > 0:
            print(
                f"[LOG] Completed in {minutes} minutes and {seconds} seconds using {self.api_calls} api calls."
            )
        else:
            print(
                f"[LOG] Completed in {seconds} seconds using {self.api_calls} api calls."
            )
