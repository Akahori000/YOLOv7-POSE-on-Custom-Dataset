
import os
import random
import shutil

def split_files(source_folder, folder1, folder2, folder3, split_ratio=(0.7, 0.1, 0.2)):
    # フォルダが存在しない場合は作成
    os.makedirs(folder1, exist_ok=True)
    os.makedirs(folder2, exist_ok=True)
    os.makedirs(folder3, exist_ok=True)

    # フォルダ内のファイルを取得
    all_files = os.listdir(source_folder)
    total_files = len(all_files)

    # ratioで分割
    split_points = [int(total_files * ratio) for ratio in split_ratio]
    
    # ファイルをランダムにシャッフル
    random.shuffle(all_files)

    # ファイルを分割してコピー
    for i, file_name in enumerate(all_files):
        source_path = os.path.join(source_folder, file_name)
        if i < split_points[0]:
            destination_path = os.path.join(folder1, file_name)
        elif i < split_points[0] + split_points[1]:
            destination_path = os.path.join(folder2, file_name)
        else:
            destination_path = os.path.join(folder3, file_name)
        shutil.copy2(source_path, destination_path)

# 例として、ソースフォルダ内のファイルを7:1:2で分割して、フォルダ1、フォルダ2、フォルダ3にコピーする
source_folder = '/workspace/data/echo/Dataset_BUSI_with_GT/data/normal'
folder1 = '/workspace/data/echo/Dataset_BUSI_with_GT/data/train/normal'
folder2 = '/workspace/data/echo/Dataset_BUSI_with_GT/data/val/normal'
folder3 = '/workspace/data/echo/Dataset_BUSI_with_GT/data/test/normal'


split_files(source_folder, folder1, folder2, folder3, split_ratio=(0.8, 0.2, 0.0))