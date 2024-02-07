import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir', default='workspace', type=str)
parser.add_argument('--gpu', default=0, type=int, help='ID of GPU to use')
args = parser.parse_args()

files = glob.glob(f'{args.dir}/*.ply')

for file in files:
    name = file.replace('.ply', '')
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python convert.py big --test_path {file}')
    # os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} kire {name}.glb --save_video {name}_mesh.mp4 --wogui')