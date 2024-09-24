import json
import math
import sys
import numpy as np
sys.path.insert(0, '..')
from VQA_models.utils import load_json

def euclidian_distance(p1, p2):
    """Calculate the euclidian distance between two points."""
    return math.sqrt(sum((p1[i] - p2[i])**2 for i in range(3)))

def get_centroid(extents):
    return [(extents[i] + extents[i + 3])/2 for i in range(3)]

def get_intersection_with_bbox(extents, centroid1, centroid2):
    """Get intersection points of a line connecting centroid1, centroid2
    with the des of a bounding box defined by its extents."""
    min_corner = extents[:3]
    max_corner = extents[3:]

    planes = [
        (min_corner, [1, 0, 0]),
        (max_corner, [1, 0, 0]),
        (min_corner, [0, 1, 0]),
        (max_corner, [0, 1, 0]),
        (min_corner, [0, 0, 1]),
        (max_corner, [0, 0, 1]),
    ]

    for plane_point, plane_normal in planes:
        intersection = line_plane_intersection(centroid1, centroid2, plane_point, plane_normal) # returns None if line and plane are parallel
        if intersection is not None:
            # check if intersection is within the bounding box (within the plane's extents)
            if all(min_corner[i] - 1e-6 <= intersection[i] <= max_corner[i] + 1e-6 for i in range(3)): # buffer for floating point errors
                return intersection

def line_plane_intersection(p0, p1, plane_point, plane_normal):
    """Calculate the intersection point of a line and a plane."""
    p0 = np.array(p0)
    p1 = np.array(p1)
    plane_point = np.array(plane_point)
    plane_normal = np.array(plane_normal)
    
    line_dir = p1 - p0
    dot = np.dot(plane_normal, line_dir)
    
    if abs(dot) < 1e-6:  # Line and plane are parallel
        return None
    
    w = plane_point - p0
    d = np.dot(plane_normal, w) / dot
    if d < 0: # Intersection is behind the line
        return None
    
    intersection = p0 + d * line_dir
    return intersection

def get_overlap_coef(bb_extents_A, bb_extents_B):
    """Calculate the overlap coefficient between two bounding boxes.
    overlap {
        = 1 : bounding box contact
        > 1 : bounding boxes overlap
        < 1 : bounding boxes are distanced (no contact, no overlap)
    """
    centroidA = get_centroid(bb_extents_A)
    centroidB = get_centroid(bb_extents_B)

    intA = get_intersection_with_bbox(bb_extents_A, centroidA, centroidB)
    intB = get_intersection_with_bbox(bb_extents_B, centroidA, centroidB)
    
    A_B = euclidian_distance(centroidA, centroidB)
    A_intA = euclidian_distance(centroidA, intA)
    B_intB = euclidian_distance(centroidB, intB)

    overlap = A_B / (A_intA + B_intB)
    return overlap

def get_volume_ratio(bb_extents_A, bb_extents_B):
    """
    Calculate the volume ratio between two bounding boxes (A to B).
    Make sure to call this function with consistent order of bounding boxes (eg. source to target ratio)
    """
    dx_A = abs(bb_extents_A[3] - bb_extents_A[0])
    dy_A = abs(bb_extents_A[4] - bb_extents_A[1])
    dz_A = abs(bb_extents_A[5] - bb_extents_A[2])
    volume_A = dx_A * dy_A * dz_A
    dx_B = abs(bb_extents_B[3] - bb_extents_B[0])
    dy_B = abs(bb_extents_B[4] - bb_extents_B[1])
    dz_B = abs(bb_extents_B[5] - bb_extents_B[2])
    volume_B = dx_B * dy_B * dz_B
    if volume_B == 0:
        volume_B = 1e-2
        # print('Target volume is zero')
    return volume_A / volume_B

def get_axis_projection_coefs(bb_extents_A, bb_extents_B):
    """
    Calculate the axis-projection coefficient between two bounding boxes.
    """
    centroidA = get_centroid(bb_extents_A)
    centroidB = get_centroid(bb_extents_B)
    A_B = euclidian_distance(centroidA, centroidB)
    x_p = (centroidA[0] - centroidB[0]) / A_B
    y_p = (centroidA[1] - centroidB[1]) / A_B
    z_p = (centroidA[2] - centroidB[2]) / A_B
    return [x_p, y_p, z_p]

def get_distance_to_floor(extents, scene_idx, node_features):
    return extents[2]
    # objects = node_features[scene_idx]
    # for obj_idx, obj in objects.items():
    #     if obj['nyu40_label'] == 2:
    #         print('found floor')
    #         z_floor = (obj['extents'][2] + obj['extents'][5]) / 2
    #         z_obj = (extents[2] + extents[5]) / 2
    #         print(z_floor)
    #         return z_obj - z_floor
    # return None

if __name__ == '__main__':
    # Test code:
    sg = load_json('/local/home/akjaer/edge_pred/ScanNet_scripts/scene0000_node_features.json')
    sg = sg['scene0000_00']
    extents_A = sg['0']['extents']
    extents_B = sg['21']['extents']
    source_label = sg['0']['label']
    target_label = sg['21']['label']

    overlap = get_overlap_coef(extents_A, extents_B)
    volume_ratio = get_volume_ratio(extents_A, extents_B)
    axis_projection_coefs = get_axis_projection_coefs(extents_A, extents_B)

    print(source_label, target_label)
    print(f"Overlap: {overlap}")
    print(f"Volume ratio: {volume_ratio}")
    print(f"Axis projection coefficients: {axis_projection_coefs}")