# labelboxのannotation結果(json)を読み込む
# 動画を読み込み、annotation結果に合わせてフレーム毎に振り分けていく


import json
import cv2
import os
import numpy as np

file_path = '/workspace/dataset/export-result.ndjson'
output_folder = "/workspace/dataset/images"

def _is_labeled(frame, list):
    pair_num = int(len(list)/2)
    flg = False
    for i in range(pair_num):
        if int(list[i*2]) <= frame <= int(list[(i*2)+1]):
            flg = True
            return flg
    return flg


## フレームの抽出
# 例：        
  # annotated_good =  ['37', '55', '78', '104'] # 開始フレーム、終了フレームの対の繰り返し
  # annotated_perfect = ['1', '36', '56', '77', '105', '183']
      
def extract_frames(video_path, output_folder, annotated_good, annotated_perfect):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    filename = os.path.basename(path)
    f_name = os.path.splitext(filename)[0]
    
    if not os.path.exists(f"{output_folder}/Good"):
        os.mkdir(f"{output_folder}/Good")
    if not os.path.exists(f"{output_folder}/Perfect"):
        os.mkdir(f"{output_folder}/Perfect")
    if not os.path.exists(f"{output_folder}/No_label"):
        os.mkdir(f"{output_folder}/No_label")

    annotation = []
    for frm in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        label = "no_label"
        frame_num = frm + 1 # LabelBoxが1からカウントしている
        if _is_labeled(frame_num, annotated_good) == True:
            print(frame_num, "good")
            label = "Good"
            annotation.append(1)
        elif _is_labeled(frame_num, annotated_perfect) == True:
            print(frame_num, "perfect")
            label = "Perfect"
            annotation.append(2)
        else:
            print(frame_num, "no_label") 
            label = "No_label"    
            annotation.append(0)          

        frame_filename = f"{output_folder}/{label}/{f_name}_frame_{str(frm + 1)}.bmp"
        # cv2.imwrite(frame_filename, frame)

    cap.release()
    print(annotation)
    labels = np.array(annotation)
    np.savetxt(f"{os.path.splitext(video_path)[0]}_label.csv", labels, fmt="%d", delimiter=",")



with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]


# どんなカテゴリがあるか抽出
category_list = []
for frame_data in data:
    projects = frame_data.get('projects', {})
    for project_id, project_data in projects.items():
        labels = project_data.get('labels', {})
        if labels:
            frames = labels[0].get('annotations', {}).get('frames', {})
            for frame_number, frame_info in frames.items():
                category = frame_info.get("classifications")[0].get("name") # categoryを読み出す
                category_list.append(category)
print(set(category_list))
categories = list(set(category_list))

# 上記で抽出できるけど、普通に定義しておく　順番変わるとややこしいので
categories = ["Good", "Perfect"]

# labelを抽出
for frame_data in data:
    projects = frame_data.get('projects', {})
    for project_id, project_data in projects.items():
        annotated_good = []
        annotated_perfect = []
        annotation_all_kind = []
        labels = project_data.get('labels', {})
        if labels:
            frames = labels[0].get('annotations', {}).get('frames', {})

            category_list = []
            for frame_number, frame_info in frames.items():
                category = frame_info.get("classifications")[0].get("name")
                if category == categories[0]:
                    annotated_good.append(frame_number)
                else:
                    annotated_perfect.append(frame_number)
        
        print("annotated_good", annotated_good)
        print("annotated_perfect", annotated_perfect)
        annotation_all_kind.append(annotated_good)
        annotation_all_kind.append(annotated_perfect)


        # video_paths = []
        path = frame_data["data_row"]["row_data"]
        print(path)
        # video_paths.append(path)
        extract_frames(path, output_folder, annotated_good=annotated_good, annotated_perfect=annotated_perfect)

