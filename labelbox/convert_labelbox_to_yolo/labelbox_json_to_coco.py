import ndjson
import json
import os
import numpy as np

category_to_id = {"Perfect": 2, "Good": 1, "Poor": 0}

def read_ndjson(file_path):
    with open(file_path, 'r') as f:
        data = ndjson.load(f)
    return data

# Specify the path to your NDJSON file
# file_path = '/workspace/shizuka_labelbox/Export v2 project - Ultrasound-Yobi-set-annotation - 2_23_2024.ndjson'
# project_id = "cls3a54bb00er07xl1ast3e0a"
file_path = "/workspace/shizuka_keypointdetection/YOLOv7-POSE-on-Custom-Dataset/labelbox/convert_labelbox_to_yolo/Export v2 project - Ultrasound-240118-set-annotation-teruya-sensei - 3_3_2024.ndjson"
project_id = "cls77p97w0z0f07v48rvy5f27"
# Read the NDJSON file
ndjson_data = read_ndjson(file_path)

# Initialize the COCO dataset structure
coco_dataset = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "ultrasound_anatomy",
            "supercategory": "body",
            "keypoints": ["Epicondyle", "intersect-ligament-humerus", "Lowest-humerus", "Right-edge-humerus", "Left-edge-ulna", "Right-of-ulna"]
        }
    ]
}

for data in ndjson_data:
    # Function to add a frame as an image in COCO format
    def add_image(frame_id, file_name, width, height):
        coco_dataset["images"].append({
            "id": frame_id,
            "width": width,
            "height": height,
            "file_name": file_name
        })

    # Function to add keypoints from a frame to COCO annotations
    def add_annotation(image_id, category_id, keypoints, annotation_id):
        coco_dataset["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "keypoints": keypoints,
            "num_keypoints": int(len(keypoints) / 3),
            "iscrowd": 0,
            "area": 0,  # Not applicable but required
            "bbox": [0, 0, 0, 0]  # Not applicable but required
        })

    # Processing the data
    annotation_id = 1  # Initialize annotation IDs
    # video name
    name = os.path.splitext(data["data_row"]["external_id"])[0]
    

    # keyの値に応じて昇順にソート
    dic = data['projects'][project_id]['labels'][0]['annotations']['frames']
    # dic2 = {k: dic[k] for k in sorted(dic.keys(), key=int)}
    keys_int = [int(k) for k in dic.keys()]        # keyを文字列→整数に
    print(np.min(keys_int), np.max(keys_int))
    
    for frame in range(np.min(keys_int), np.max(keys_int)+1, 1):
        frame_id = "{:04d}".format(frame)
        name = os.path.splitext(data["data_row"]["external_id"])[0]
        frame_int_id = f"{name}_{frame_id}" 

        # ----lableboxのアノテーションがある場合はそのまま従う
        if frame in keys_int:

            # Assuming each frame corresponds to a separate image
            add_image(frame_int_id, f"{frame_id}.jpg", 576, 528)
            
            
            # Extract keypoints for each annotated object in the frame
            keypoints = []
            for obj_id, obj_data in dic[str(frame)]["objects"].items():
                for keypoint_name in coco_dataset["categories"][0]["keypoints"]:
                    # Assuming a placeholder for missing keypoints (x=0, y=0, v=0)
                    x, y, v = 0, 0, 0
                    if obj_data['name'] == keypoint_name:
                        x, y = obj_data['point']['x'], obj_data['point']['y']
                        v = 2  # Assuming the keypoint is visible and labeled
                        keypoints.extend([x, y, v])
            
            # Extract category_id
            if dic[str(frame)]["classifications"] != []:
                category = dic[str(frame)]["classifications"][0]["radio_answer"]["name"]
                category_id = category_to_id.get(category, 0)
            else:
                category_id = coco_dataset["annotations"][-1]["category_id"] # Same as the previous frame
                    
            if keypoints != []:
                add_annotation(frame_int_id, category_id, keypoints, annotation_id)
                annotation_id += 1
            else:
                add_annotation(frame_int_id, category_id, keypoints, annotation_id)
                annotation_id += 1

        # ------labelboxのアノテーションがない場合は空の値を突っ込む
        else:
            # Extract category_id
            category_id = coco_dataset["annotations"][-1]["category_id"]  # Same as the previous frame
            if frame == 1:  # アノテーションしていない場合に備えて
                category_id = 0 

            # Assuming each frame corresponds to a separate image
            add_image(frame_int_id, f"{frame_id}.jpg", 576, 528)
            keypoints = []
            add_annotation(frame_int_id, category_id, keypoints, annotation_id)
            annotation_id += 1


# Output the COCO formatted data to a JSON file
with open('/workspace/shizuka_keypointdetection/YOLOv7-POSE-on-Custom-Dataset/labelbox/convert_labelbox_to_yolo/coco_all_info_0118set.json', 'w') as f:
    json.dump(coco_dataset, f, indent=4)

print("Conversion to COCO format completed.")