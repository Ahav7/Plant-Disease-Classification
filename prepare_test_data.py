"""
Script to move a few images from the original dataset
into the test dataset folder randomly.
"""

import shutil
import glob
import random
import os
from tqdm import tqdm
from class_names import class_names as CLASS_NAMES

random.seed(42)

ROOT_DIR = os.path.join('', 'plantvillage dataset', 'color')
DEST_DIR = os.path.join('', 'input', 'test')
# Class directories.
class_dirs = CLASS_NAMES
# Test images.
test_split = 0.2

for class_dir in class_dirs:
    os.makedirs(os.path.join(DEST_DIR, class_dir), exist_ok=True)
    init_image_paths = glob.glob(os.path.join(ROOT_DIR, class_dir, "*"))
    print(f"Initial number of images for class {class_dir}: {len(init_image_paths)}")
    random.shuffle(init_image_paths)
    test_images = random.sample(init_image_paths, int(round(test_split*len(init_image_paths))))
    print(f"Copying {len(test_images)} images from {class_dir}")
    for test_image_path in tqdm(test_images):
        image_name = test_image_path.split(os.path.sep)[-1]
        shutil.move(test_image_path, os.path.join(DEST_DIR, class_dir, image_name))
    final_image_paths = glob.glob(os.path.join(ROOT_DIR, class_dir, '*'))
    print(f"Final number of images for class {class_dir}: {len(final_image_paths)}\n")
