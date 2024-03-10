import ndjson
import json
import os
import numpy as np
import cv2
import pandas as pd


category_to_id = {"Perfect": 2, "Good": 1, "Poor": 0}

# 1動画における各フレームのcategory取得
def get_category_id(dic, keys_int):
    category_ids = []
    for frame in range(np.min(keys_int), np.max(keys_int)+1, 1):
        frame_id = str(frame)

        # ------lableboxのアノテーションがある場合はそのまま従う
        if frame in keys_int:
            # Extract category_id
            if dic[frame_id]["classifications"] != []:
                category = dic[frame_id]["classifications"][0]["radio_answer"]["name"]
                category_id = category_to_id.get(category, 0)
            else:
                category_id = category_ids[-1] # Same as the previous frame
        
        # ------labelboxのアノテーションがない場合は前の値を入れる
        else:
            # Extract category_id
            category_id = category_ids[-1]  # Same as the previous frame
        
        category_ids.append(category_id)  
    
    return(category_ids)

# mp4を読み込み
def convert_mp4_to_bmp(video_name, input_folder, output_folder, category_id):
    # Output folderを作成
    os.makedirs(output_folder, exist_ok=True)

    # input_folder内の全mp4ファイルに対して処理
    filename = video_name + ".mp4"
    if filename.endswith(".mp4"):
        input_path = os.path.join(input_folder, filename)
        
        # VideoCaptureを作成
        cap = cv2.VideoCapture(input_path)
        # フレーム数を取得
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        filename = os.path.basename(input_path)

        # フレームごとに処理
        for frame_num in range(total_frames):
            # フレームを読み込む
            ret, frame = cap.read()
            if not ret:
                break

            # ファイル名を構築
            # if category_id[frame_num] == 1:
            #     save_path = f"{output_folder}/Good/"
            # elif category_id[frame_num] == 2:
            #     save_path = f"{output_folder}/Perfect/"
            # else:
            #     save_path = f"{output_folder}/Poor/"
            save_path = f"{output_folder}/"

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            output_filename = f"{os.path.splitext(filename)[0]}_{(frame_num+1):04d}.bmp"
            output_path = os.path.join(save_path, output_filename)

            # BMP形式で保存
            cv2.imwrite(output_path, frame)

        # VideoCaptureを解放
        cap.release()

# jsonファイルの読み込み
def read_ndjson(file_path):
    with open(file_path, 'r') as f:
        data = ndjson.load(f)
    return data


# Specify the path to your NDJSON file
file_path = '/workspace/shizuka_labelbox/Export v2 project - Ultrasound-Yobi-set-annotation - 2_23_2024.ndjson'

#file_path = '/workspace/shizuka_labelbox/Export v2 project - Ultrasound-Yobi-set-annotation - 2_23_2024.ndjson'
# project_id = "cls3a54bb00er07xl1ast3e0a"
file_path = "/workspace/shizuka_keypointdetection/YOLOv7-POSE-on-Custom-Dataset/labelbox/convert_labelbox_to_yolo/Export v2 project - Ultrasound-240118-set-annotation-teruya-sensei - 3_3_2024.ndjson"
project_id = "cls77p97w0z0f07v48rvy5f27"


# input_folder_path = '/workspace/data/ultrasound/dataset/mp4_1'
input_folder_path = "/workspace/shizuka_keypointdetection/YOLOv7-POSE-on-Custom-Dataset/labelbox/data/mp4_1"
# output_folder_path = '/workspace/data/ultrasound/dataset/bmp_categorized_by_labelbox_anno'
output_folder_path = "/workspace/shizuka_keypointdetection/YOLOv7-POSE-on-Custom-Dataset/labelbox/data/bmp_1"

files = os.listdir(input_folder_path)

for file in files:    # mp4ごと
    # labelboxの情報を取り出す
    # dic = data['projects'][project_id]['labels'][0]['annotations']['frames']
    # keys_int = [int(k) for k in dic.keys()]        # keyを文字列→整数に
    # print(np.min(keys_int), np.max(keys_int))      # keys_int : labelboxに格納されてるラベル

    # video_name = os.path.splitext(data["data_row"]["external_id"])[0]
    video_name = os.path.splitext(os.path.basename(file))[0]

    # フレーム毎にcategory_idを取得
    category_ids = 0 #get_category_id(dic, keys_int)

    # mp4→bmp変換
    convert_mp4_to_bmp(video_name, input_folder_path, output_folder_path, category_ids)


print(".")