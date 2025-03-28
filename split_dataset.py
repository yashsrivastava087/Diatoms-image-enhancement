import os
import shutil

input_dir = 'diatomic_dataset/resized_images'
train_dir = 'diatomic_dataset/train'
val_dir = 'diatomic_dataset/val'
test_dir = 'diatomic_dataset/test'

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

train_ratio = 0.8
val_ratio = 0.1

filenames = os.listdir(input_dir)
num_files = len(filenames)
train_num = int(num_files * train_ratio)
val_num = int(num_files * val_ratio)

for i, filename in enumerate(filenames):
    src_path = os.path.join(input_dir, filename)
    if i < train_num:
        shutil.copy(src_path, train_dir)
    elif i < train_num + val_num:
        shutil.copy(src_path, val_dir)
    else:
        shutil.copy(src_path, test_dir)
