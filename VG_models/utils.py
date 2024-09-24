import json
import PIL.Image
import numpy as np
import torch
import clip
import os
import random
import wandb
from configparser import ConfigParser

# ---------------------------------
# Constants
# ---------------------------------
# Corners of a unit cube
bb_corners = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
    [0, 1, 1]
]

# ---------------------------------
# Data loading and writing
# ---------------------------------
def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def load_image_array(image_path):
    image = PIL.Image.open(image_path)
    image = np.array(image)
    return image

def load_txt(path, separator: str = ' '):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [list(map(float, line.strip().split(separator))) for line in lines] # Remove newline characters and split each line into a list of floats
    return np.array(lines)

def write_json(data, path, indent=4):
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)
    print(f'Saved data to {path}')
    return

def load_config(config_path='./config.yaml'):
    config = ConfigParser()
    config.read(config_path)
    return config

# ---------------------------------
# Seed
# ---------------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}.")


# ---------------------------------
# Logging and checkpoints
# ---------------------------------
def wandb_logging(config, mode: str, data={}, epoch=None, verbose=True):
    assert mode in ['init', 'log'], "Select mode from ['init', 'log']"

    if mode == 'init':
        if eval(config['wandb']['log']) == True:
            wandb.init(project=config['wandb']['project_name'], name=config['wandb']['run_name'])
            wandb.config.update(config)
    
    elif mode == 'log':
        assert epoch != None, "Specify current epoch"
        if epoch % 2 == 0 and eval(config['wandb']['log']):
            if verbose: print('Logging results to WandB...')
            wandb.log(data)

def save_ckpt(config, epoch, model, verbose=True):
    if epoch % 5 == 0 and eval(config['model']['save_model']):
        run_name = config['wandb']['run_name']
        torch.save(model.state_dict(), f'/cluster/scratch/akjaer/checkpoints/{run_name}_ckpt_{epoch}.pth')
        if verbose: print(f"Model saved to /cluster/scratch/akjaer/checkpoints/{run_name}_ckpt_{epoch}.pth")

# ---------------------------------
# Transforms
# ---------------------------------
def T_world2screen(x_world, intrinsics, pose):
    x_world_ = np.hstack((x_world,1))
    x_cam_ = np.linalg.inv(pose) @ x_world_
    x_screen_ = intrinsics @ x_cam_
    x_screen = np.round(x_screen_[:2] / x_screen_[2]).astype(int)
    return x_screen

# ---------------------------------
# CLIP
# ---------------------------------
def clip_embed(x, type="image"):
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    if type == "image":
        image = preprocess(PIL.Image.fromarray(x)).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(image)
    elif type == "text":
        text = clip.tokenize(x).to(device)
        with torch.no_grad():
            features = model.encode_text(text)
    
    return features

def clip_compare(image, text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(PIL.Image.fromarray(image)).unsqueeze(0).to(device)
    text = clip.tokenize(text).to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
    return logits_per_image, logits_per_text

# ---------------------------------
# Geometry
# ---------------------------------
def euclidean_distance(p0, p1):
    return np.linalg.norm(np.array(p0) - np.array(p1))

def line_plane_intersection(p0, p1, plane_point, plane_normal):
    """Calculate the intersection point of a line and a plane."""
    p0 = np.array(p0)
    p1 = np.array(p1)
    plane_point = np.array(plane_point)
    plane_normal = np.array(plane_normal)
    
    line_dir = p1 - p0
    dot = np.dot(plane_normal, line_dir)
    
    if abs(dot) < 1e-6:  # Line and plane are parallel (or quasi-parallel)
        return None
    
    w = plane_point - p0
    d = np.dot(plane_normal, w) / dot
    if d < 0: # Intersection is behind the line origin
        return None
    
    intersection = p0 + d * line_dir
    return intersection

def get_intersection_with_bbox(extents_source, p0, p1):
    """
    Get intersection point of a line connecting points p0 and p1
    with the intersected face of the source bounding box.
    """
    min_corner = extents_source[0]
    max_corner = extents_source[1]

    planes = [
        (min_corner, [1, 0, 0]),
        (max_corner, [1, 0, 0]),
        (min_corner, [0, 1, 0]),
        (max_corner, [0, 1, 0]),
        (min_corner, [0, 0, 1]),
        (max_corner, [0, 0, 1]),
    ]

    for plane_point, plane_normal in planes:
        intersection = line_plane_intersection(p0, p1, plane_point, plane_normal) # returns None if line and plane are parallel
        if intersection is not None:
            # check if intersection is within the bounding box (within the plane's extents)
            if all(min_corner[i] - 1e-6 <= intersection[i] <= max_corner[i] + 1e-6 for i in range(3)): # buffer for floating point errors
                return intersection

def AABB_min_distance(extents0, extents1):
    A_min, A_max = map(np.array, extents0)
    B_min, B_max = map(np.array, extents1)

    delta1 = A_min - B_max
    delta2 = B_min - A_max
    u = np.max(np.array([np.zeros(len(delta1)), delta1]), axis=0)
    v = np.max(np.array([np.zeros(len(delta2)), delta2]), axis=0)
    dist = np.linalg.norm(np.concatenate([u, v]))
    return dist
            
# ---------------------------------
# Graphs
# ---------------------------------
def adj_matrix2list(adj_mat):
    adj_list = []
    for i, row in enumerate(adj_mat):
        for j, col in enumerate(row):
            if col == 1: # If there is an edge between node i and node j
                adj_list.append([i, j])
    return adj_list

