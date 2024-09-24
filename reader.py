import argparse
import os
import sys

from SensorData import SensorData

# params
parser = argparse.ArgumentParser()
# paths
parser.add_argument('--split_file', required=True, help='path to the split file')
parser.add_argument('--input_base_path', help='base path to input folders', default='/cluster/project/cvg/data/scannet/scans/')
parser.add_argument('--output_base_path', help='base path to output folders', default='/cluster/scratch/akjaer/Datasets/ScanNet/scans/')
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
parser.add_argument('--export_poses', dest='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False)

opt = parser.parse_args()
print(opt)

def process_scene(scene_folder):
    input_file = os.path.join(opt.input_base_path, scene_folder, f'{scene_folder}.sens')
    output_path = os.path.join(opt.output_base_path, scene_folder)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # load the data
    print(f'loading {input_file}...')
    sd = SensorData(input_file)
    print('loaded!')
    
    if opt.export_depth_images:
        sd.export_depth_images(os.path.join(output_path, 'depth'))
    if opt.export_color_images:
        sd.export_color_images(os.path.join(output_path, 'color'))
    if opt.export_poses:
        sd.export_poses(os.path.join(output_path, 'pose'))
    if opt.export_intrinsics:
        sd.export_intrinsics(os.path.join(output_path, 'intrinsic'))

def main():
    with open(opt.split_file, 'r') as file:
        for line in file:
            scene_folder = line.strip()
            process_scene(scene_folder)

if __name__ == '__main__':
    main()
