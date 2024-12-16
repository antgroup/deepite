import json
import csv
import os.path
import torch.nn.functional as F

import numpy as np
import torch
import pandas as pd

from utils.preprocessing import create_adjacency_matrix
from utils.preprocessing import standardize, normalize
from torch.utils.data import Dataset

class Synthetic_Dataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = []
        self.labels = []
        self.graphs = []

        for subdir, _, files in os.walk(dataset_path):
            if 'cases' in subdir:
                cases_dir = subdir
                graph_path = os.path.join(os.path.dirname(cases_dir[:-5]), 'graph.json')

                with open(graph_path, 'r') as f:
                    graph = json.load(f)
                self.graphs.append(graph)

                # Load all case data and info
                for case_dir in os.listdir(cases_dir):
                    case_path = os.path.join(cases_dir, case_dir)
                    if not os.path.isdir(case_path):
                        continue 

                    data_path = os.path.join(case_path, 'data.csv')
                    info_path = os.path.join(case_path, 'info.json')

                    df = pd.read_csv(data_path, header=None)
                    data = df.values

                    with open(info_path, 'r') as f:
                        info = json.load(f)

                    num_samples = df.shape[0]
                    num_causes = len(info['causes'])
                    labels = np.zeros((num_samples, data.shape[-1]), dtype=int)
                    for i in range(info['length_normal'], num_samples):
                        for j, cause in enumerate(info['causes']):
                            labels[i, cause] = 1  # Mark as anomaly for each cause

                    self.data.append(data)
                    self.labels.append(labels)

        all_data = np.concatenate(self.data, axis=0).reshape(-1, self.data[0].shape[-1])
        self.mean = np.mean(all_data, axis=0)
        self.std = np.std(all_data, axis=0)
        self.std[self.std == 0] = 1
        self.min_val = np.min(all_data, axis=0)
        self.max_val = np.max(all_data, axis=0)
        self.max_val[self.max_val == self.min_val] = 1 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        graph = self.graphs[idx]

        features = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        graph_adj = torch.tensor(create_adjacency_matrix(graph), dtype=torch.long)

        features = standardize(features, self.mean, self.std)
        features = normalize(features, self.min_val, self.max_val)

        return {'X': features, 'Y': labels, 'adj': graph_adj}

class SPGC_Dataset(Dataset):
    def __init__(self, dataset_path):
        features_path = os.path.join(dataset_path, 'data.csv')
        labels_path = os.path.join(dataset_path, 'label.csv')
        graph_path = os.path.join(dataset_path, 'graph.json')
        self.features = pd.read_csv(features_path)
        self.labels = pd.read_csv(labels_path)
        with open(graph_path, 'r') as f:
            self.graph = json.load(f)
        self.graph_adj = create_adjacency_matrix(self.graph)

        self.data = pd.merge(self.features, self.labels, on='sample_index')
        self.data = self.data.to_dict(orient='records')

        all_features = np.array([[sample[f'feature{i}'] for i in range(3)] + [sample[f'feature{i}_1'] for i in range(3, 4)] + [
            sample[f'feature{i}_2'] for i in range(3, 4)] + [sample[f'feature{i}_3'] for i in range(3, 4)] + [
                       sample[f'feature{i}_4'] for i in range(3, 4)] + [sample[f'feature{i}'] for i in
                                                                        range(11, 20)] + [sample['feature60']] + [sample['feature20_distance'],
                                                                                          sample['featureY_if'],
                                                                                          sample['featureX_if'],
                                                                                          sample['featureY_mean'],
                                                                                          sample['featureX_mean'],
                                                                                          sample['featureY_min'],
                                                                                          sample['featureX_min']] for sample in self.data])
        
        all_features = np.nan_to_num(all_features, nan=0)
        
        self.mean = np.mean(all_features, axis=0)
        self.std = np.std(all_features, axis=0)
        self.std[self.std == 0] = 1 
        self.min_val = np.min(all_features, axis=0)
        self.max_val = np.max(all_features, axis=0)
        self.max_val[self.max_val == self.min_val] = 1 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        features = [sample[f'feature{i}'] for i in range(3)] + [sample[f'feature{i}_1'] for i in range(3, 4)] + [
            sample[f'feature{i}_2'] for i in range(3, 4)] + [sample[f'feature{i}_3'] for i in range(3, 4)] + [
                       sample[f'feature{i}_4'] for i in range(3, 4)] + [sample[f'feature{i}'] for i in
                                                                        range(11, 20)] + [sample['feature60']] + [sample['feature20_distance'],
                                                                                          sample['featureY_if'],
                                                                                          sample['featureX_if'],
                                                                                          sample['featureY_mean'],
                                                                                          sample['featureX_mean'],
                                                                                          sample['featureY_min'],
                                                                                          sample['featureX_min']]

        features = [0 if x is None or np.isnan(x) else x for x in features]

        labels = [0] * 24
        if sample['causes_type'] == 'rootcause1':
            labels[9] = 1  # feature13
            labels[11] = 1  # feature15

        elif sample['causes_type'] == 'rootcause2':
            labels[15] = 1  # feature19
            labels[18] = 1  # featureY_if
            labels[19] = 1  # featureX_if
            labels[20] = 1
            labels[21] = 1  
            labels[22] = 1
            labels[23] = 1

        elif sample['causes_type'] == 'rootcause3':
            labels[16] = 1  # feature60
            labels[17] = 1  # feature20_distance
            labels[18] = 1  # featureY_if
            labels[19] = 1  # featureX_if
            labels[20] = 1  
            labels[21] = 1
            labels[22] = 1
            labels[23] = 1  

        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        features = standardize(features, self.mean, self.std)
        features = normalize(features, self.min_val, self.max_val)

        return {'X': features, 'Y': labels, 'adj': self.graph_adj}

class Protein_Dataset(Dataset):
    def __init__(self, dataset_path):
        features_path = os.path.join(dataset_path, 'data.csv')
        graph_path = os.path.join(dataset_path, 'graph.json')
        self.features = pd.read_csv(features_path)
        self.data = self.features.values
        with open(graph_path, 'r') as f:
            self.graph = json.load(f)
        self.graph_adj = create_adjacency_matrix(self.graph)
        all_data = np.concatenate(self.data[:,:-2], axis=0).reshape(-1, self.data[:,:-2].shape[-1])
        self.mean = np.mean(all_data, axis=0)
        self.std = np.std(all_data, axis=0)
        self.std[self.std == 0] = 1
        self.min_val = np.min(all_data, axis=0)
        self.max_val = np.max(all_data, axis=0)
        self.max_val[self.max_val == self.min_val] = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        features = sample[:-2]
        labels = sample[-1]

        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        graph_adj = torch.tensor(self.graph_adj, dtype=torch.long)

        features = standardize(features, self.mean, self.std)
        features = normalize(features, self.min_val, self.max_val)

        num_classes = len(features)
        if labels.item() >= num_classes:
            labels = torch.zeros(num_classes, dtype=torch.float32)
        else:
            labels = F.one_hot(labels, num_classes=num_classes).float()

        return {'X': features, 'Y': labels, 'adj': graph_adj}

def load_data(dataset_type, dataset_path):
    if dataset_type == "Synthetic":
        return Synthetic_Dataset(dataset_path)
    elif dataset_type == "Protein":
        return Protein_Dataset(dataset_path)
    elif dataset_type == "SPGC":
        return SPGC_Dataset(dataset_path)
    else:
        raise ValueError(f"Dataset Type {dataset_type} not Supported!")