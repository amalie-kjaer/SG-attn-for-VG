import os
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Dataset, Data
from transformers import CLIPTokenizer, CLIPTextModel

import sys
sys.path.append('..')
from utils import adj_matrix2list, load_json, write_json

class ScanNetPyG(Dataset): #TODO: Inherit from ScanGraphQA dataset
    def __init__(
        self,
        root,
        split_file_path,
        cropping_method,
        qa_dataset,
        verbose=True,
    ):

        self.cropping_method = cropping_method
        self.root = root

        all_qa = load_json(qa_dataset)
        with open(split_file_path, 'r') as f:
            lines = f.readlines()
        self.split_file = [line.strip().split(' ')[0] for line in lines]

        self.qa = [qa for qa in all_qa if qa['scene_id'] in self.split_file]

        if verbose:
            print(f'{len(self.qa)} questions, {len(self.split_file)} scenes in dataset.')

        super().__init__(root)

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'last_test') #TODO
    
    @property
    def processed_file_names(self):
        os.makedirs(os.path.dirname(self.processed_dir), exist_ok=True)
        return [os.path.join(self.processed_dir, f'{scene_id}-{self.cropping_method}.pt') for scene_id in self.split_file]

    def process(self):
        for scene_id in self.split_file:
            
            nodes_clip = load_json(os.path.join(self.root, 'ScanNet/scans/', scene_id, f'{scene_id}_clip_features{self.cropping_method}.json'))
            nodes_ft = load_json(os.path.join(self.root, 'ScanNet/scans/', scene_id, f'{scene_id}_node_features.json'))
            labels_ft = load_json(os.path.join(self.root, 'ScanNet/scans/', scene_id, f'{scene_id}_label_features.json'))
            # adj = load_json(os.path.join(self.root, 'ScanNet/scans/', scene_id, f'{scene_id}_adjacency_matrix_0.5.json'))
            adj = load_json(os.path.join(self.root, 'ScanNet/scans/', scene_id, f'{scene_id}_adjacency_matrix.json'))
            
            keys_sorted = [id for id in nodes_ft.keys()]
            # x = [nodes_clip[str(key)] for key in keys_sorted] # order of x must match order of nodes_ft and adjacency matrix
            
            # For toy experiment: 
            x = [labels_ft[str(key)][0] for key in keys_sorted]
            x = torch.tensor(x, dtype=torch.float)

            adj_list = adj_matrix2list(adj)
            edge_index = torch.tensor(adj_list, dtype=torch.long)
            edge_index = edge_index.t().contiguous()
            
            # keys_sorted = torch.tensor([int(k) for k in keys_sorted], dtype=torch.long) # Store IDs (node ID may not be the same as index in x: depends how ScanNet's segmentation JSON file was made)
            data = Data(x=x, edge_index=edge_index)
            torch.save(data, os.path.join(self.processed_dir, f'{scene_id}-{self.cropping_method}.pt'))
   
    def len(self):
        return len(self.qa)
    
    def get(self, idx):
        question = self.qa[idx]['question']
        scene_id = self.qa[idx]['scene_id']
        graph = torch.load(os.path.join(self.processed_dir, f'{scene_id}-{self.cropping_method}.pt'))
        answer_label = self.qa[idx]['object_names'][0]
        answer_instance = self.qa[idx]['object_ids'][0]
        long_answer = self.qa[idx]['answers'][0]

        return graph, question, answer_instance, answer_label, scene_id

if __name__ == "__main__":

    dataset = ScanNetPyG(
        root='/cluster/scratch/akjaer/Datasets/',
        split_file_path='/cluster/scratch/akjaer/split_files/QA_all_scenes.txt',
        cropping_method="110_viz",
        qa_dataset="/cluster/scratch/akjaer/Datasets/ScanQA/ScanQA_clean_all.json"
    )

    # print(dataset.processed_file_names)
    print(len(dataset.qa))
    # scene_id = "scene0000_00"
    # nodes = load_json(os.path.join('/cluster/scratch/akjaer/Datasets/ScanNet/scans/', scene_id, f'{scene_id}_clip_features.json'))

    # x = [(k, v[:3]) for k, v in nodes.items()]
    # print(x)


    # for i in range(len(dataset)):
    #     data, question_raw = dataset[i]
    #     print(data)
    #     print(question_raw)
    #     print(data.y)
    #     print()
    
    # print('done')