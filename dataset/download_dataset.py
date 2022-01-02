import os
import requests
import zipfile


def create_dir(dir_name):
    try:
        os.mkdir(dir_name)
    except Exception as e:
        print(e)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, destination)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def _save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def unzip_to_dataset(zip_path, dataset_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)


def import_dataset(dataset_info: dict):
    _import_dataset(dataset_info['zip_id'], dataset_info['zip_name'],
                    dataset_info['csv_id'], dataset_info['csv_name'],
                    dataset_info['destination_path'])


def _import_dataset(zip_id, zip_name, csv_id, csv_name, destination_path=''):
    if destination_path != '' and not os.path.exists(destination_path):
        create_dir(destination_path)

    # download zip images data
    print(f'downloading {zip_name}')
    if destination_path == '':
        zip_destination = zip_name
    else:
        zip_destination = f'{destination_path}/{zip_name}'
    download_file_from_google_drive(zip_id, zip_destination)

    # download label csv
    print(f'downloading {csv_name}')
    if destination_path == '':
        csv_destination = csv_name
    else:
        csv_destination = f'{destination_path}/{csv_name}'
    download_file_from_google_drive(csv_id, csv_destination)

    # unzip images data
    print(f'unziping {zip_destination}')
    unzip_to_dataset(zip_destination, destination_path)


# function demo
if __name__ == '__main__':
    pokemon_dataset = {
        'zip_id': '1-MYHHJqFIbmiAKD1PQP7sQKFkjkRX1Lx',
        'zip_name': 'pokemonclassification.zip',
        'csv_id': '1PdF1ZlhfOo_l5co_CKabTj1P0ZamT6Q1',
        'csv_name': 'pokemon_dataset.csv',
        'destination_path': ''
    }

    # import_dataset(pokemon_dataset)
