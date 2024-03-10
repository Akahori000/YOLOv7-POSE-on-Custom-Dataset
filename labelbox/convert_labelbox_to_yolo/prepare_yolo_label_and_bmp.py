import ndjson
import json
import os
import numpy as np
import cv2
import pandas as pd
import shutil

combinations = [
    # combination 0
    {
        "train": ["00000004.mp4", "00000006.mp4", "00000015.mp4", "00000010.mp4", "00000012.mp4", "00000018.mp4"],
        "val": ["00000005.mp4"],
        "test": ["000000011.mp4"]
    },
    # combination 1
    {
        "train": ["00000006.mp4", "00000010.mp4", "00000011.mp4", "00000012.mp4", "00000015.mp4", "00000018.mp4"],
        "val": ["00000004.mp4"],
        "test": ["000000005.mp4"]
    },
    # combination 2 (assuming you meant 2 instead of a second 3)
    {
        "train": ["00000004.mp4", "000000005.mp4", "00000010.mp4", "00000012.mp4", "00000015.mp4", "00000018.mp4"],
        "val": ["00000011.mp4"],
        "test": ["000000006.mp4"]
    }
]

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

# mp4を読み込んでクラスごとにbmp保存
def convert_mp4_to_bmp(video_name, input_folder, output_folder, category_id):
    # Output folderを作成
    os.makedirs(output_folder, exist_ok=True)

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
            if category_id[frame_num] == 1:
                save_path = f"{output_folder}/Good/"
            elif category_id[frame_num] == 2:
                save_path = f"{output_folder}/Perfect/"
            else:
                save_path = f"{output_folder}/Poor/"

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            output_filename = f"{os.path.splitext(filename)[0]}_{(frame_num+1):04d}.bmp"
            output_path = os.path.join(save_path, output_filename)

            # BMP形式で保存
            cv2.imwrite(output_path, frame)

        # VideoCaptureを解放
        cap.release()

# 画像とyoloのtxtを一気に保存
def yolo_save_bmp_and_txt(frame, frame_num, filename, save_path, txt_save_path, yolo_txt_folder):
    output_filename = f"{os.path.splitext(filename)[0]}_{(frame_num+1):04d}.bmp"
    output_path = os.path.join(save_path, output_filename)
    cv2.imwrite(output_path, frame)
    # txtを保存
    txt_output_filename = f"{os.path.splitext(filename)[0]}_{(frame_num+1):04d}.txt"
    txt_input_path = f"{yolo_txt_folder}/{txt_output_filename}"
    txt_output_path = f"{txt_save_path}/{txt_output_filename}"
    shutil.copy(txt_input_path, txt_output_path)

# Train/valは Good, perfect画像のみで行う
# Testは Good, Perfect,Poor全部で行う
def train_data_save(frame, output_folder, yolo_txt_folder, category_id, filename, frame_num):
    #  Good/Perfect クラスは 学習に用いる
    if category_id[frame_num] == 1 or  category_id[frame_num] == 2:     # Good or Perfect
        save_path =     f"{output_folder}/images/train/"
        txt_save_path = f"{output_folder}/labels/train/"
        yolo_save_bmp_and_txt(frame, frame_num, filename, save_path, txt_save_path, yolo_txt_folder)
    

def val_data_save(frame, output_folder, yolo_txt_folder, category_id, filename, frame_num):
    # Good/Perfect クラスはvalidationに用いる
    if category_id[frame_num] == 1 or category_id[frame_num] == 2:      # Good or Perfect
        save_path = f"{output_folder}/images/val/"
        txt_save_path = f"{output_folder}/labels/val/"
        yolo_save_bmp_and_txt(frame, frame_num, filename, save_path, txt_save_path, yolo_txt_folder)


def test_data_save(frame, output_folder, yolo_txt_folder, category_id, filename, frame_num):
    # category_idによらずGood/Perfect/Poor 全クラスをTestに用いる
    save_path = f"{output_folder}/images/test/"
    txt_save_path = f"{output_folder}/labels/test/"

    yolo_save_bmp_and_txt(frame, frame_num, filename, save_path, txt_save_path, yolo_txt_folder)
    

# prepare_yolo_label_and_bmp
def prepare_yolo_label_and_bmp(video_name, input_folder, yolo_txt_folder, output_folder, category_id, combination):
    # Output folderを作成
    os.makedirs(output_folder, exist_ok=True)
    filename = video_name + ".mp4"

    # log
    if filename in combination["train"]:
        print("train:", filename)
    elif filename in combination["val"]:
        print("val:", filename)
    else:
        print("test:", filename)

    # mp4の読み込み
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

            # 動画.mp4がtrain/val/testのいずれかによって保存の仕方が変わる
            if filename in combination["train"]:
                train_data_save(frame, output_folder, yolo_txt_folder, category_id, filename, frame_num)
            elif filename in combination["val"]:
                val_data_save(frame, output_folder, yolo_txt_folder, category_id, filename, frame_num)
            else:
                test_data_save(frame, output_folder, yolo_txt_folder, category_id, filename, frame_num)

        # VideoCaptureを解放
        cap.release()

# jsonファイルの読み込み
def read_ndjson(file_path):
    with open(file_path, 'r') as f:
        data = ndjson.load(f)
    return data


# Specify the path to your NDJSON file
file_path = '/workspace/shizuka_labelbox/Export v2 project - Ultrasound-Yobi-set-annotation - 2_23_2024.ndjson'
# Read the NDJSON file
ndjson_data = read_ndjson(file_path)


for data in ndjson_data:    # mp4ごと
    # labelboxの情報を取り出す
    dic = data['projects']['cls3a54bb00er07xl1ast3e0a']['labels'][0]['annotations']['frames']
    keys_int = [int(k) for k in dic.keys()]        # keyを文字列→整数に
    # print("frames:", np.min(keys_int), "-", np.max(keys_int))      # keys_int : labelboxに格納されてるラベル

    video_name = os.path.splitext(data["data_row"]["external_id"])[0]
    
    # フレーム毎にcategory_idを取得
    category_ids = get_category_id(dic, keys_int)

    input_folder_path = '/workspace/data/ultrasound/dataset/mp4_1'
    # mp4→bmp変換
    # output_folder_path = '/workspace/data/ultrasound/dataset/bmp_categorized_by_labelbox_anno'
    # convert_mp4_to_bmp(video_name, input_folder_path, output_folder_path, category_ids)    

    # yolo用にlabelとbmpを準備
    comb_num = 2
    yolo_txt_folder =  '/workspace/data/ultrasound/dataset/yolo_all_frames'
    output_yolo_folder = "/workspace/shizuka_keypointdetection/YOLOv7-POSE-on-Custom-Dataset/final_dataset/combination" + str(comb_num)
    for dir in ["test", "train", "val"]:
        os.makedirs(f"{output_yolo_folder}/images/{dir}", exist_ok=True)
        os.makedirs(f"{output_yolo_folder}/labels/{dir}", exist_ok=True)

    prepare_yolo_label_and_bmp(video_name, input_folder_path, yolo_txt_folder, output_yolo_folder, category_ids, combinations[comb_num])

print(".")