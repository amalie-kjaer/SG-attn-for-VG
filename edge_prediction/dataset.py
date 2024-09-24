import numpy as np
import os
import json
import torch
import torch_geometric
import math
import random
import sys
from utils_feature_extraction import *

sys.path.insert(0, '..')
from VQA_models.utils import load_json, one_hot, map_label_to_int

class SG():
    def __init__(self, split=None, transform=None, augmentation_prob=0.7, norm=True):
        self.norm = norm
        self.augmentation_prob = augmentation_prob
        self.split = split

        if self.split == 'train': # Train on 3RScan dataset (because we have GT labels for all relations)
            self.node_features = load_json('/local/home/akjaer/edge_pred/3RScan_scripts/node_feature_dataset.json')
            self.edge_labels = load_json('/local/home/akjaer/edge_pred/3RScan_scripts/edge_label_dataset_NEW.json')
        elif self.split == 'test': # Test on ScanNet
            self.node_features = load_json('/local/home/akjaer/edge_pred/ScanNet_scripts/scene0000_node_features.json')
            self.edge_labels = load_json('/local/home/akjaer/edge_pred/ScanNet_scripts/scene0000_edge_labels.json')
        
        self.x = []
        self.y = []
        # Loop through relations and append to relevant node features to x and relation label to y
        for scene_idx, relations_list in self.edge_labels.items():
            for r in relations_list:
                source_idx = r[0]
                target_idx = r[1]
                relation = r[2]

                source_nyu_label = int(self.node_features[scene_idx][str(source_idx)]['nyu40_label'])
                target_nyu_label = int(self.node_features[scene_idx][str(target_idx)]['nyu40_label'])
                
                source_extents = self.node_features[scene_idx][str(source_idx)]['extents']
                target_extents = self.node_features[scene_idx][str(target_idx)]['extents']

                overlap = get_overlap_coef(source_extents, target_extents)
                volume_ratio = get_volume_ratio(source_extents, target_extents)
                axis_projection_coefs = get_axis_projection_coefs(source_extents, target_extents)
                distance_to_floor = get_distance_to_floor(source_extents, scene_idx, self.node_features)

                self.x.append(np.concatenate((overlap, axis_projection_coefs, volume_ratio, source_nyu_label, target_nyu_label), axis=None))
                self.y.append(int(map_label_to_int[relation]))
        
        # Compute mean and std for nomralization
        self.mean = np.mean(self.x, axis=0)
        self.mean[-2:] = 0.0
        self.std = np.std(self.x, axis=0)
        self.std[-2:] = 1.0

        print(f'Finished loading {self.split} dataset. Number of samples: {len(self.x)}.')

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        X = self.x[idx]

        # Normalize data
        if self.norm:
            X = self.normalize_data(X)
        # Augment data (with probability augmentation_prob)
        if random.random() < self.augmentation_prob:
            X = self.augment_data(X)
        y = self.y[idx]

        return X, y
    
    def normalize_data(self, X):
        return (X - self.mean) / self.std
    
    def augment_data(self, X):
        return X

if __name__ == '__main__':
    dataset = SG(split='train')
    print(dataset[0])