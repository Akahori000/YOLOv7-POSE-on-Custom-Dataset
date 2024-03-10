import ndjson
import json
import os

def read_ndjson(file_path):
    with open(file_path, 'r') as f:
        data = ndjson.load(f)
    return data

# Specify the path to your NDJSON file
file_path = '/workspace/shizuka_labelbox/Export v2 project - Ultrasound-Yobi-set-annotation - 2_12_2024.ndjson'

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

    for frame_id, frame_data in data['projects']['cls3a54bb00er07xl1ast3e0a']['labels'][0]['annotations']['frames'].items():
        # Convert frame_id to int for consistent ID management
        name = os.path.splitext(data["data_row"]["external_id"])[0]
        frame_int_id = f"{name}_{frame_id}" #int(frame_id)
        
        # Assuming each frame corresponds to a separate image
        add_image(frame_int_id, f"{frame_id}.jpg", 576, 528)
        
        
        # Extract keypoints for each annotated object in the frame
        keypoints = []
        for obj_id, obj_data in frame_data['objects'].items():
            for keypoint_name in coco_dataset["categories"][0]["keypoints"]:
                # Assuming a placeholder for missing keypoints (x=0, y=0, v=0)
                x, y, v = 0, 0, 0
                if obj_data['name'] == keypoint_name:
                    x, y = obj_data['point']['x'], obj_data['point']['y']
                    v = 2  # Assuming the keypoint is visible and labeled
                    keypoints.extend([x, y, v])
        
        if keypoints != []:
            add_annotation(frame_int_id, 1, keypoints, annotation_id)
            annotation_id += 1

# Output the COCO formatted data to a JSON file
with open('/workspace/shizuka_labelbox/coco_format_all_in_1line.json', 'w') as f:
    json.dump(coco_dataset, f, indent=4)

print("Conversion to COCO format completed.")