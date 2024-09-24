import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
from operator import add
from plyfile import PlyData
import pandas as pd
import shutil
import torch
import sys
import cv2
import PIL.Image
from utils import *
from transformers import CLIPModel, AutoProcessor

# Dataset class for a single scene
class Scene:
    def __init__(self, dataset_path, output_path, scene_id, scanqa_dataset, model, processor, verbose=True, light_version=True):
        self.dataset_path = dataset_path
        self.scanqa_dataset = scanqa_dataset
        self.output_path = output_path
        self.scene_id = scene_id
        self.verbose = verbose
        self.light_version=light_version
        self.model = model
        self.processor = processor
        
        # Check arguments
        self.__check_data_exists()

        # Lazy loading of variables
        self._frames_in_scene = None
        self._segmentation = None
        self._vertex_segments = None
        self._vertex_coords = None
        self._label_conversion = None
        self._label_features = None
        self._node_features = None
        self._adjacency_matrix = None
        self._rgb_images = None
        self._segmentation_per_frame = None
        self._camera_poses = None
        self._intrinsics = None
        self._avg_clip_features = None
        self._qa = None
        self._dataset_vocabulary = None

    def __check_data_exists(self):
        path = os.path.join(self.dataset_path, f'scannet/scans/{self.scene_id}')
        assert Path(path).exists(), f"Download directory {path} does not exist."
        # TODO: also check that scanqa_dataset path is ok.
    
    # -------------------------------------------
    # Extract scene data for scene_id
    # -------------------------------------------
    @property
    def frames_in_scene(self):
        if self._frames_in_scene is None:
            c_frames = os.listdir(os.path.join(self.dataset_path, f'{self.output_path}/{self.scene_id}/color/'))
            all_frames = [f.split('.')[0] for f in c_frames]
            if self.light_version:
                self._frames_in_scene = all_frames[::2] # ::10 for sim, ::2 for viz
            else:
                self._frames_in_scene = all_frames
        return self._frames_in_scene

    @property
    def n_frames(self):
        return len(self.frames_in_scene)

    @property
    def segmentation(self):
        if self._segmentation is None:
            self._segmentation = load_json(os.path.join(self.dataset_path, 'scannet/scans', self.scene_id, self.scene_id + '_vh_clean.aggregation.json'))
        return self._segmentation
    
    @property
    def vertex_segments(self):
        if self._vertex_segments is None:
            self._vertex_segments = load_json(os.path.join(self.dataset_path, 'scannet/scans', self.scene_id, self.scene_id + '_vh_clean_2.0.010000.segs.json'))
        return self._vertex_segments

    @property
    def vertex_coords(self):
        if self._vertex_coords is None:
            plydata = PlyData.read(os.path.join(self.dataset_path, 'scannet/scans', self.scene_id, self.scene_id + '_vh_clean_2.ply'))
            self._vertex_coords = np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']), axis=-1)
        return self._vertex_coords

    @property
    def label_conversion(self):
        if self._label_conversion is None:
            self._label_conversion = pd.read_csv(os.path.join(self.dataset_path, 'scannet/scannetv2-labels.combined.tsv'), sep = '\t')
        return self._label_conversion

    @property
    def rgb_images(self):
        if self._rgb_images is None:
            self._rgb_images = [load_image_array(os.path.join(self.output_path, f'{self.scene_id}/color/{frame}.jpg')) for frame in self.frames_in_scene]
            if self.verbose:
                print(f'Loaded {len(self._rgb_images)} RGB images for {self.scene_id}.')
        return self._rgb_images
    
    @property
    def segmentation_per_frame(self):
        if self._segmentation_per_frame is None:
            self._segmentation_per_frame = [load_image_array(os.path.join(self.output_path, f'{self.scene_id}/instance-filt/{frame}.png')) for frame in self.frames_in_scene]
            if self.verbose:
                print(f'Loaded {len(self._segmentation_per_frame)} instance segmentation masks for {self.scene_id}.')
        return self._segmentation_per_frame
    
    @property
    def camera_poses(self):
        if self._camera_poses is None:
            self._camera_poses = [load_txt(os.path.join(self.output_path, f'{self.scene_id}/pose/{frame}.txt')) for frame in self.frames_in_scene]
            if self.verbose:
                print(f'Loaded {len(self._camera_poses)} camera poses for {self.scene_id}.')
        return self._camera_poses
    
    @property
    def intrinsics(self):
        if self._intrinsics is None:
            self._intrinsics = load_txt(os.path.join(self.output_path, f'{self.scene_id}/intrinsic/intrinsic_color.txt'))
        return self._intrinsics
    
    @property
    def dataset_vocabulary(self):
        if self._dataset_vocabulary is None:
            self._dataset_vocabulary = load_json('./dataset_vocab.json')
        return self._dataset_vocabulary

    # -------------------------------------------
    # Construct scene graph (node features and proximity adjacency matrix) for scene_id
    # -------------------------------------------
    def node_features(self, force_recompute=False):
        if self._node_features is None:
            path = os.path.join(self.output_path, f'{self.scene_id}/{self.scene_id}_node_features.json')
            if os.path.exists(path) and not force_recompute:
                self._node_features = load_json(path)
                if self.verbose:
                    print(f'Loaded node features for {self.scene_id} from {path}.')
            else:
                self._node_features = self.__get_node_features()
        return self._node_features
    
    def __get_node_features(self):
        def segmentID_to_vertexID(segmentIDs, vertex_segments):
            return np.where(np.isin(vertex_segments['segIndices'], segmentIDs))[0]

        node_features = {}
        for obj in self.segmentation['segGroups']:
            obj_id = str(obj['id'])
            points = self.vertex_coords[segmentID_to_vertexID(obj['segments'], self.vertex_segments)]
            min_coords, max_coords = np.min(points, axis=0), np.max(points, axis=0)
            centroid = (min_coords + max_coords) / 2
            node_features[obj_id] = {
                'label': obj['label'],
                'nyu40_label': int(self.label_conversion[self.label_conversion['raw_category'] == obj['label']]['nyu40id'].values[0]),
                'extents': [min_coords.tolist(), max_coords.tolist()],
                'centroid': centroid.tolist(),
            }
        output_path = os.path.join(self.output_path, f'{self.scene_id}/{self.scene_id}_node_features.json')
        write_json(node_features, output_path)
        return node_features
    
    def proximity_adjacency_matrix(self, threshold: float = 0.2, force_recompute=False):
        if self._adjacency_matrix is None:
            path = os.path.join(self.output_path, f'{self.scene_id}/{self.scene_id}_adjacency_matrix.json')
            if os.path.exists(path) and not force_recompute:
                self._adjacency_matrix = load_json(path)
                if self.verbose:
                    print(f'Loaded adjacency matrix for {self.scene_id} from {path}.')
            else:
                self._adjacency_matrix = self.__get_proximity_adjacency_matrix(threshold)
        return self._adjacency_matrix

    def __get_proximity_adjacency_matrix(self, threshold):
        num_objects = len(self.segmentation['segGroups'])
        adjacency_matrix = np.zeros((num_objects, num_objects))

        for i, (obj_id_s, obj_features_s) in enumerate(self.node_features().items()):
            for j, (obj_id_t, obj_features_t) in enumerate(self.node_features().items()):
                if i < j: # only fill the upper triangle of adj matrix
                    if AABB_min_distance(obj_features_s['extents'], obj_features_t['extents']) <= threshold:
                        adjacency_matrix[int(obj_id_s), int(obj_id_t)] = 1

        adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T) # make adjacency matrix symmetric
        np.fill_diagonal(adjacency_matrix, 1) # add self-loops

        output_path = os.path.join(self.output_path, f'{self.scene_id}/{self.scene_id}_adjacency_matrix.json')
        write_json(adjacency_matrix.tolist(), output_path)
    
    @property
    def generate_random_adjacency(self):
        num_objects = len(self.segmentation['segGroups'])
        adjacency_matrix = np.zeros((num_objects, num_objects))
        # Fill the upper triangle of the matrix with random 0.0 or 1.0
        for i in range(num_objects):
            for j in range(i+1, num_objects):
                random_value = np.random.choice([0.0, 1.0])
                adjacency_matrix[i, j] = random_value

        # Make the matrix symmetric by copying the upper triangle to the lower triangle
        adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)

        # Fill the diagonal with 1.0 to represent self-loops
        np.fill_diagonal(adjacency_matrix, 1.0)
        write_json(adjacency_matrix.tolist(), os.path.join(self.output_path, f'{self.scene_id}/{self.scene_id}_adjacency_matrix_random.json'))


    # -------------------------------------------
    # Extract QA-pairs for scene_id
    # -------------------------------------------
    @property
    def qa(self):
        if self._qa is None:
            full_qa_data = load_json(os.path.join(self.dataset_path, f'scanqa/{self.scanqa_dataset}'))
            qa_data = [item for item in full_qa_data if item['scene_id'] == self.scene_id]
            self._qa = [(x['question'], x['object_ids']) for x in qa_data]
        return self._qa
    
    # -------------------------------------------
    # Compute k-average CLIP embeddings for objects in scene_id
    # -------------------------------------------
    def kavg_clip_features(self, k: int, cropping_method: int, scoring: str, force_recompute=False, preview=False):
        # assert scoring in ['viz', 'sim', 'vizW1', 'vizW2', 'vizW0'], "Select scoring method from ['viz', 'sim', 'vizW1', 'vizW2', 'vizW0'] (highest visibility score or highest clip-similarity to object label)"
        if self._avg_clip_features is None:
            path = os.path.join(self.output_path, f'{self.scene_id}/{self.scene_id}_clip_features{cropping_method}{k}_{scoring}.json')
            # Load k-average CLIP embeddings from file if it exists and not forced to recompute
            if os.path.exists(path) and not force_recompute:
                self._avg_clip_features = load_json(path)
                if self.verbose:
                    print(f'Loaded k-average CLIP embeddings for {self.scene_id} from {path}.')
            # Compute k-average CLIP embeddings
            else:
                self._avg_clip_features = self.__get_kavg_clip_features(k, cropping_method, scoring, preview)
        return self._avg_clip_features
    
    def __get_kavg_clip_features(self, k, cropping_method, scoring, preview):
        """
        Return: WEIGHTED k-average CLIP embedding for cropped images of each object in the scene.
        The weight is proportional to the visibility score of the object in the frame.
        """
        k_weighted_clip_features = {}
        total_weight = {}

        print('Retrieving best image views for each object in the scene: locate each object, calculate visibility score, select frames with best visibility score.')
        top_k_frames_per_object = self.__top_k_frames_per_object(k, scoring)
        
        # print(top_k_frames_per_object)

        print(f'Calculating average CLIP embeddings for each object. Cropping method: {cropping_method}, k: {k}, scoring method: {scoring}')
        for obj_id, frame_score_tuple in tqdm(top_k_frames_per_object.items()):
            for frame, score in frame_score_tuple:
                # Crop image around object
                crop = self.__crop_image_around_object(frame, obj_id, cropping_method, preview)
                # Get CLIP embedding of the cropped image
                inputs = self.processor(images=crop, return_tensors="pt")
                image_features = self.model.get_image_features(**inputs)
                
                # Add to sum of all running embeddings to dictionary
                if obj_id not in k_weighted_clip_features:
                    k_weighted_clip_features[obj_id] = [x * score for x in image_features[0].tolist()]
                    total_weight[obj_id] = score
                else:
                    k_weighted_clip_features[obj_id] = list(map(add, k_weighted_clip_features[obj_id], [x * score for x in image_features[0].tolist()]))
                    total_weight[obj_id] += score
    
        # Calculate weighted average
        kavg_clip_features = {obj_id: [x / total_weight[obj_id] for x in k_weighted_clip_features[obj_id]] for obj_id in k_weighted_clip_features}
        output_path = os.path.join(self.output_path, f'{self.scene_id}/{self.scene_id}_clip_features{cropping_method}{k}_{scoring}.json')
        write_json(kavg_clip_features, output_path)
        return kavg_clip_features
    
    def __top_k_frames_per_object(self, k, scoring):
        """
        Input: k (int), the number of frames to return for each object in the scene.
        Returns: dictionary of the top k frames with the highest visibility scores for each object in a given scene.
        The frames are sorted by highest visibility score.
        Used to calculate the k-average CLIP embeddings for each object in the scene.
        Example output:
        {
            obj_id_1: [(frame_id_top1, viz_score_top1), (frame_id_top2, viz_score_top2), ...],
            obj_id_2: [(frame_id_top1, viz_score_top1), (frame_id_top2, viz_score_top2), ...],
            ...
        }
        """
        all_object_scores = {}
        for frame_id in tqdm(self.frames_in_scene):
            # Calculate visibility scores for all objects in the frame
            if scoring == 'viz':
                wp, wc = 1, 1
                frame_scores = self.__viz_scores_in_frame(frame_id, wp, wc)
            elif scoring == 'vizHF':
                wp, wc = 1, 0.5
                frame_scores = self.__viz_scores_in_frame(frame_id, wp, wc)
            elif scoring == 'vizW3':
                wp, wc = 0, 100
                frame_scores = self.__viz_scores_in_frame(frame_id, wp, wc)
            elif scoring == 'sim':
                frame_scores = self.__similarity_scores_in_frame(frame_id)

            # Add the visibility scores from each frame to the dictionary
            for obj_id, score in frame_scores.items():
                if obj_id not in all_object_scores:
                    all_object_scores[obj_id] = [(frame_id, score)]
                else:
                    all_object_scores[obj_id].append((frame_id, score))
            
        # Sort the visibility scores for each object in descending order
        for obj_id, scores in all_object_scores.items():
            scores.sort(key=lambda x: x[1], reverse=True)
            top_k_frames = {int(ins): scores[:k] for ins, scores in all_object_scores.items()}

        return top_k_frames
    
    def __viz_scores_in_frame(self, frame_id:str, wp, wc):
        """
        Input: frame_id (str)
        Returns: dictionary of the visibility scores for all objects in a given frame.
        viz_score = pixel_score * corner_score * 100, with:
            pixel_score = number of pixels occupied by the object / total number of pixels in the frame
            corner_score = number of bounding box corners of the object visible in the frame / total number of bounding box corners
        Objects that are fully visible in the frame and occupy a large portion of the frame will have high visibility scores.
        Example output:
        {
            obj_id_1: viz_score_1,
            obj_id_2: viz_score_2,
            ...
        }
        """
        # Load instance segmentation
        ins_array = self.segmentation_per_frame[self.frames_in_scene.index(frame_id)]
        objects_in_frame = [x - 1 for x in np.unique(ins_array) if x != 0] # objects are 0-indexed while instance ids are 1-indexed
        
        # Load RGB image
        rgb_array = self.rgb_images[self.frames_in_scene.index(frame_id)]
        height_pixels, width_pixels = rgb_array.shape[:2]

        # Load camera pose in current frame
        pose = self.camera_poses[self.frames_in_scene.index(frame_id)]

        # Loop through all objects present in the frame
        viz_scores = {}
        for obj_id in objects_in_frame:
            # ------- Calculate pixel score ------- #
            obj_pixels = np.where(ins_array == obj_id + 1)
            pixel_score = len(obj_pixels[0]) / (height_pixels * width_pixels)

            # ------- Calculate corner score ------- #
            extents = self.node_features()[str(obj_id)]['extents']
            corners_world = [[extents[x][0], extents[y][1], extents[z][2]] for x, y, z in bb_corners]
            
            corner_score = 0
            for c_world in corners_world:
                c_screen = T_world2screen(c_world, self.intrinsics, pose)
                if 0 <= c_screen[0] < width_pixels and 0 <= c_screen[1] < height_pixels:
                    corner_score += 1
            corner_score /= 8

            # ------- Calculate viz score ------- #
            viz_score = pixel_score * corner_score # TODO: could also add a weight to pixel_score and corner_score
            # viz_score = corner_score # TODO: could also add a weight to pixel_score and corner_score
            viz_scores[obj_id] = viz_score
        
        return viz_scores
      
    def __similarity_scores_in_frame(self, frame_id:str):
        # Load instance segmentation
        ins_array = self.segmentation_per_frame[self.frames_in_scene.index(frame_id)]
        objects_in_frame = [x - 1 for x in np.unique(ins_array) if x != 0] # objects are 0-indexed while instance ids are 1-indexed
        
        # Load RGB image
        rgb_array = self.rgb_images[self.frames_in_scene.index(frame_id)]
        
        # Loop through all objects present in the frame
        scores = {}
        for obj_id in objects_in_frame:
            obj_pixels = np.where(ins_array == obj_id + 1)
            label = self.node_features()[str(obj_id)]['label']
            
            # Square crop around object, and resize
            x, y = np.where(ins_array == int(obj_id) + 1)
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
    
            height, width, _ = rgb_array.shape
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            crop_size = int(max(y_max - y_min, x_max - x_min))

            crop_x_min = max(0, center_x - crop_size // 2)
            crop_x_max = min(center_x + crop_size // 2, height)
            crop_y_min = max(0, center_y - crop_size // 2)
            crop_y_max = min(center_y + crop_size // 2, width)

            rgb_cropped = rgb_array[crop_x_min:crop_x_max, crop_y_min:crop_y_max, :]
            rgb_cropped = cv2.resize(rgb_cropped, (224, 224))

            # Get normalized CLIP embedding of the cropped image and image label
            crop_features = clip_embed(rgb_cropped, type="image")
            crop_features = crop_features / crop_features.norm(dim=1, keepdim=True)
            label_features = clip_embed(label, type="text")
            label_features = label_features / label_features.norm(dim=1, keepdim=True)

            # Compute similarity score
            similarity_score = (crop_features @ label_features.T).item()
            scores[obj_id] = similarity_score

        return scores

    def __crop_image_around_object(self, frame_id, obj_id, cropping_method, preview):
        """
        Cropping methods:
        - 0: Crop the image exactly around the object's bounding box (no padding) and delete the rest of the image (surrounding and occluding pixels are set to 0)
        - 1: Crop the image exactly around the object's bounding box (no padding) and leave the rest of the image (surrounding and occluding pixels are part of the crop)
        - 2: Square crop + resize to 224x224 images
        """
        assert cropping_method in [0, 1, 2, 3], "Invalid cropping method. Choose from [0, 1, 2]."

        rgb_array = self.rgb_images[self.frames_in_scene.index(frame_id)]
        ins_array = self.segmentation_per_frame[self.frames_in_scene.index(frame_id)]

        if cropping_method == 0:
            # only keep object pixels
            x, y = np.where(ins_array == int(obj_id) + 1)
            empty_img = np.zeros_like(rgb_array)
            empty_img[x, y] = rgb_array[x, y]
            
            # crop around object
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            rgb_cropped = empty_img[x_min:x_max, y_min:y_max, :]

        elif cropping_method == 1:
            x, y = np.where(ins_array == int(obj_id) + 1)
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            rgb_cropped = rgb_array[x_min:x_max, y_min:y_max, :]

        elif cropping_method == 2:
            x, y = np.where(ins_array == int(obj_id) + 1)
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)

            height, width, _ = rgb_array.shape
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            crop_size = int(max(y_max - y_min, x_max - x_min))

            crop_x_min = max(0, center_x - crop_size // 2)
            crop_x_max = min(center_x + crop_size // 2, height)
            crop_y_min = max(0, center_y - crop_size // 2)
            crop_y_max = min(center_y + crop_size // 2, width)

            rgb_cropped = rgb_array[crop_x_min:crop_x_max, crop_y_min:crop_y_max, :]
            rgb_cropped = cv2.resize(rgb_cropped, (224, 224))
        
        elif cropping_method == 3:
            x, y = np.where(ins_array == int(obj_id) + 1)
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            rgb_cropped = rgb_array[x_min:x_max, y_min:y_max, :]
            rgb_cropped = cv2.resize(rgb_cropped, (224, 224))

        if preview:
            rgb_cropped_image = PIL.Image.fromarray(rgb_cropped)
            rgb_cropped_image.save(f'./preview/{self.scene_id}_{obj_id}_{frame_id}.png')
            print(f"Saved crop to ./preview/{self.scene_id}_{obj_id}_{frame_id}.png")
        
        return rgb_cropped

    # -------------------------------------------
    # Features for toy example: CLIP(label)
    # -------------------------------------------
    @property
    def label_features(self):
        if self._label_features is None:
            path = os.path.join(self.output_path, f'{self.scene_id}/{self.scene_id}_label_features.json')
            if os.path.exists(path):
                self._label_features = load_json(path)
                if self.verbose:
                    print(f'Loaded label features for {self.scene_id} from {path}.')
            else:
                self._label_features = self.__get_label_features()
        return self._label_features
    
    def __get_label_features(self):
        label_features = {}
        for obj in self.segmentation['segGroups']:
            obj_id = str(obj['id'])
            label = obj['label']
            label_features[obj_id] = clip_embed(label, type="text").tolist()
        write_json(label_features, os.path.join(self.output_path, f'{self.scene_id}/{self.scene_id}_label_features.json'))

    # -------------------------------------------
    # Compare text query to object CLIP embeddings
    # TODO: Accomodate for multiple questions (matrix multiplication)
    # -------------------------------------------
    def query_similarity(self, query: str):
        avg_clip_features = self.kavg_clip_features()
        # print('avg_clip_features', avg_clip_features.keys())
        # print('avg_clip_features', avg_clip_features['9'][-3:])
        
        # Extract CLIP embeddings from avg_clip_features dictionnary
        instance_clip = []
        for ins in avg_clip_features:
            instance_clip.append(torch.tensor([avg_clip_features[ins]]))
        # print(len(instance_clip)) # 67
        # print(instance_clip[1].shape) # torch.Size([1, 512])
        # print(instance_clip[1][0].shape) # torch.Size([512])
        
        instance_clip = torch.cat(instance_clip)
        # print(instance_clip.shape) # torch.Size([67, 512])

        # Get CLIP embedding for the query
        query_features = clip_embed(query, type="text")
        # print(query_features.shape) # torch.Size([1, 512])

        # Normalize the embeddings
        instance_clip /= instance_clip.norm(dim=0, keepdim=True)
        # print(instance_clip.norm(dim=0, keepdim=True).shape) # torch.Size([1, 512])
        # print(instance_clip.shape) # torch.Size([67, 512])

        query_features /= query_features.norm(dim=0, keepdim=True)
        # print(query_features.norm(dim=0, keepdim=True).shape) # torch.Size([1, 512])
        # print(query_features.shape) # torch.Size([1, 512])

        # Compute the cosine similarity between the query and the instances
        preds = (instance_clip @ query_features.T).softmax(dim=0)
        # print((instance_clip @ query_features.T).shape) # torch.Size([67, 1])
        # print(preds.shape) # torch.Size([67, 1])

        # Extract prediction
        for i, j in enumerate(avg_clip_features.keys()):
            if i == torch.argmax(preds).cpu().detach().numpy():
                predicted_instance = j
                prediction_probability = torch.max(preds).cpu().detach().numpy()
                # print(f'Predicted instance: {predicted_instance} with probability {prediction_probability}')
        
        return int(predicted_instance), prediction_probability

# Dataset class for the entire ScanGraphQA dataset
class ScanGraphQA:
    def __init__(self, dataset_path, output_path, split_file_path, scanqa_dataset):
        self.split_file_path = split_file_path
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.scanqa_dataset = scanqa_dataset

        with open(self.split_file_path, 'r') as f:
            lines = f.readlines()
        self.split_file = [line.strip().split(' ')[0] for line in lines]
        
        print('Loading CLIP model...')
        clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        print('Loading CLIP processor...')
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.scenes = [Scene(self.dataset_path, self.output_path, scene_id, self.scanqa_dataset, clip_model, processor) for scene_id in self.split_file]

    def __getitem__(self, index):
        if isinstance(index, int):
            if index < 0:
                index += len(self.scenes)
            if index >= len(self.scenes):
                raise IndexError("Index out of range")
            return self.scenes[index]
        else:
            raise TypeError("Invalid Argument Type")

    def __len__(self):
        return len(self.scenes)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_file", required=True)
    
    args = parser.parse_args()

    output_path = '/cluster/scratch/akjaer/Datasets/ScanNet/scans/'
    
    dataset = ScanGraphQA(
        dataset_path='/cluster/project/cvg/data/',
        output_path=output_path,
        split_file_path=args.split_file,
        scanqa_dataset=''
    )

    for scene in dataset:
        try:
            # ------- Load / calculate "node features" + adjacency matrix (by proximity threshold only) -------
            scene.node_features()
            scene.proximity_adjacency_matrix()           
            # ------- Load / calculate CLIP embeddings of nodes------- 
            scene.kavg_clip_features(
                cropping_method=1, 
                k=10, 
                scoring='viz', 
                force_recompute=True,
                preview=False
            )
            # ------- Remove files (needed if low storage quota): -------
            # shutil.rmtree(os.path.join(output_path, scene.scene_id, 'color'), ignore_errors=True)
            # shutil.rmtree(os.path.join(output_path, scene.scene_id, 'pose'), ignore_errors=True)
            # shutil.rmtree(os.path.join(output_path, scene.scene_id, 'intrinsic'), ignore_errors=True)
            # shutil.rmtree(os.path.join(output_path, scene.scene_id, 'instance-filt'), ignore_errors=True)
        
        except:
            print(f'{scene.scene_id} failed')