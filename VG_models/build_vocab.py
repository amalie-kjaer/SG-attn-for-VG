import os
from tqdm import tqdm
import json
from utils import load_json, write_json
from tqdm import tqdm

dataset_path = '/cluster/scratch/akjaer/Datasets/ScanNet/scans'
split_file_path = '/cluster/scratch/akjaer/split_files/QA_all_scenes.txt'

with open(split_file_path, 'r') as f:
    lines = f.readlines()
split_file = [line.strip().split(' ')[0] for line in lines]

vocab = {}
for scene_id in tqdm(split_file):
    vocab[scene_id] = {}
    node_features = load_json(os.path.join(dataset_path, scene_id, f'{scene_id}_node_features.json'))
    for node_id, node in node_features.items():
        label = node['label']
        vocab[scene_id][node_id] = label

def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'Saved data to {path}')

write_json(vocab, 'vocab_all_scenes.json')