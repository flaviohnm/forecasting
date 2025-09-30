import requests
import os

def download_all_datasets(config: dict):
    """
    Itera sobre a lista de datasets na configuração e baixa cada um.
    """
    if 'datasets' not in config or not config['datasets']:
        print("Nenhum dataset definido em './config/main_config.yaml'.")
        return

    for dataset_config in config['datasets']:
        raw_data_path = config['data_paths']['raw']
        file_path = os.path.join(raw_data_path, dataset_config['filename'])

        if os.path.exists(file_path):
            print(f"Arquivo '{dataset_config['filename']}' já existe. Download pulado.")
            continue

        print(f"Baixando dataset '{dataset_config['name']}' de {dataset_config['url']}...")
        try:
            response = requests.get(dataset_config['url'], timeout=30)
            response.raise_for_status()

            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Download de '{dataset_config['filename']}' concluído.")

        except requests.exceptions.RequestException as e:
            print(f"Falha no download de '{dataset_config['filename']}': {e}")