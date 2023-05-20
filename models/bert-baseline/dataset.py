import os
import random
import sys

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import numpy as np
import pandas as pd
import torch.utils.data as Data
from config import *
from transformers import BertTokenizer
from transformers import AutoTokenizer

message_tokenizer = BertTokenizer.from_pretrained(message_model_name)
code_tokenizer = AutoTokenizer.from_pretrained(code_model_name)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        super().__init__()
        self.messages = []
        self.codes = []
        self.labels = []
        for message, code, label in data_list:
            self.messages.append(
                message_tokenizer(message, padding='max_length', max_length=512, truncation=True, return_tensors="pt"))

            lines = code.split('\n')
            added_lines = []
            deleted_lines = []
            for line in lines:
                if line.startswith('+') and not line.startswith('+++'):
                    added_lines.append(line[1:].strip())
                elif line.startswith('-') and not line.startswith('---'):
                    deleted_lines.append(line[1:].strip())
            added_str = ' '.join(added_lines).strip()
            deleted_str = ' '.join(deleted_lines).strip()
            # print('add', added_str)
            # print('delete', deleted_str)
            self.codes.append(
                code_tokenizer(text=added_str, text_pair=deleted_str, padding='max_length', max_length=512,
                               truncation=True, return_tensors="pt"))
            self.labels.append(label)

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_message(self, idx):
        return self.messages[idx]

    def get_batch_code(self, idx):
        return self.codes[idx]

    def __getitem__(self, idx):
        batch_message = self.get_batch_message(idx)
        batch_code = self.get_batch_code(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_message, batch_code, batch_y


def read_data():
    non_vuln_data_list = []
    vuln_data_list = []

    for root, dirs, files in os.walk(DATA_PATH):
        for file_name in files:
            if file_name.endswith('.csv'):
                data = pd.read_csv(os.path.join(root, file_name))
                data = data.reset_index(drop=True)
                if file_name == 'qemu.csv':
                    data = data.drop(data.columns[0], axis=1)
                for index, row in data.iterrows():
                    assert int(row[2]) == 0 or int(row[2]) == 1
                    if int(row[2]) == 0:
                        non_vuln_data_list.append((str(row[0]), str(row[1]), int(row[2])))
                    else:
                        vuln_data_list.append((str(row[0]), str(row[1]), int(row[2])))

    return non_vuln_data_list, vuln_data_list


def split_data(non_vuln_data_list, vuln_data_list):
    random.shuffle(non_vuln_data_list)

    train_size = int(len(non_vuln_data_list) * 1.0 / 10 * 7)
    valid_size = int(len(non_vuln_data_list) * 1.0 / 10 * 2)
    test_size = len(non_vuln_data_list) - train_size - valid_size

    non_vuln_train_data = non_vuln_data_list[:train_size]
    non_vuln_valid_data = non_vuln_data_list[train_size:train_size + valid_size]
    non_vuln_test_data = non_vuln_data_list[train_size + valid_size:]

    assert len(non_vuln_train_data) + len(non_vuln_valid_data) + len(non_vuln_test_data) == len(non_vuln_data_list)

    random.shuffle(vuln_data_list)
    train_size = int(len(vuln_data_list) * 1.0 / 10 * 7)
    valid_size = int(len(vuln_data_list) * 1.0 / 10 * 2)
    test_size = len(vuln_data_list) - train_size - valid_size

    vuln_train_data = vuln_data_list[:train_size]
    vuln_valid_data = vuln_data_list[train_size:train_size + valid_size]
    vuln_test_data = vuln_data_list[train_size + valid_size:]

    assert len(vuln_train_data) + len(vuln_valid_data) + len(vuln_test_data) == len(vuln_data_list)

    train_data = non_vuln_train_data + vuln_train_data
    valid_data = non_vuln_valid_data + vuln_valid_data
    test_data = non_vuln_test_data + vuln_test_data

    return train_data, valid_data, test_data


def get_loader():
    non_vuln_data_list, vuln_data_list = read_data()
    non_vuln_data_list = non_vuln_data_list[:3000]
    vuln_data_list = vuln_data_list[:3000]
    print(len(non_vuln_data_list), len(vuln_data_list))
    train_data, valid_data, test_data = split_data(non_vuln_data_list, vuln_data_list)
    train_dataset = Dataset(train_data)
    valid_dataset = Dataset(valid_data)
    test_dataset = Dataset(test_data)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = get_loader()
    for message, code, label in train_loader:
        print(message['input_ids'].shape, message['attention_mask'].shape)
        print(code['input_ids'].shape, code['attention_mask'].shape)
        # print(code.shape)
        print(label.shape)
        # print(message['input_ids'])
        break
