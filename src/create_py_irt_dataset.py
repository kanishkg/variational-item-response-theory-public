import os
import json

from src.datasets import *

def create_pyirt_dataset(dataset, out_dir, file_name):
    responses = dataset.response.squeeze()
    with open(os.path.join(out_dir, file_name), 'w') as f:
        for i in range(responses.shape[0]):
            dict_line = {"subject_id": f"{i}", "response": f"{responses[i]}"}
            f.write(str(responses[i]) + '\n')

if __name__ == "__main__":
    # create py-irt dataset from vibo dataset
    dataset_name = 'algebraai'
    out_dir = os.path.join(DATA_DIR,'py_irt')
    train_dataset = load_dataset(dataset_name, train=True)
    test_dataset = load_dataset(dataset_name, train=False)
    create_pyirt_dataset(train_dataset, out_dir, f'{dataset_name}_train.json')
    create_pyirt_dataset(test_dataset, out_dir, f'{dataset_name}_test.json')
