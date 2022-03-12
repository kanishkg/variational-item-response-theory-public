import os
import json

from sklearn.decomposition import dict_learning

from src.datasets import *

def create_pyirt_dataset(dataset, out_dir, file_name):
    responses = dataset.response.squeeze()
    mask = dataset.mask.squeeze()
    data = []
    for i in range(responses.shape[0]):
        resp_dict = {f"{int(j)}":f"{int(responses[i,j])}" for j in range(responses.shape[1]) if mask[i,j] == 1}
        dict_line = {"subject_id": f"{int(i)}", "responses": resp_dict}
        data.append(dict_line)

    with open(os.path.join(out_dir, file_name), 'w') as f:
        f.write('\n'.join(json.dumps(i) for i in data))

if __name__ == "__main__":
    # create py-irt dataset from vibo dataset
    dataset_name = 'json'
    out_dir = os.path.join(DATA_DIR,'py_irt')
    train_dataset = load_dataset(dataset_name, train=True)
    test_dataset = load_dataset(dataset_name, train=False)
    create_pyirt_dataset(train_dataset, out_dir, f'{dataset_name}_train.json')
    create_pyirt_dataset(test_dataset, out_dir, f'{dataset_name}_test.json')
