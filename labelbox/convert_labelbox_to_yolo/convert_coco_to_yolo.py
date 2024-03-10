# 0 indicates not visible and not labeled
# 1 indicates not visible but labeled
# 2 indicates visible and labeled

import numpy as np
import json
import os

MARGIN = 30
POINT_NUM = 6

def convert_coco_to_yolo(coco_json_path, output_dir):
    # Load COCO JSON file
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each image in the dataset
    for image_data in coco_data['images']:
        image_id = image_data['id']
        image_name = image_data['file_name']
        image_width = image_data['width']
        image_height = image_data['height']
        keypoints_list = []
        category_id = []
        # Find annotations for the current image
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_id:
                keypoints = annotation['keypoints']
                keypoints_list.append(keypoints)
                category_id = annotation["category_id"]

        # Skip images without annotations
        if not keypoints_list:
            continue

        # Create YOLO annotation file
        annotation_file_name = image_id + '.txt'
        annotation_file_path = os.path.join(output_dir, annotation_file_name)
        with open(annotation_file_path, 'w') as f:
            for keypoints in keypoints_list:
                keypoints = np.array(keypoints)

                # keypointが格納されている場合
                if len(keypoints) == (POINT_NUM*3):
                    xs = keypoints[0::3]
                    ys = keypoints[1::3]
                    top_left = [np.min(xs), np.min(ys)]
                    bottom_right = [np.max(xs), np.max(ys)]
                    
                    # x_center, y_center, _ = keypoints[keypoints > 0]
                    # Normalize bounding box coordinates to range [0, 1]

                    # margins left/right/top/bottom of the bbox
                    margin_x1 = min(15, np.min(xs))                                                     # top left x margin
                    margin_x2 = 15 if (np.max(xs) + 15) < image_width else (image_width - np.max(xs))   # bottom right x margin
                    margin_y1 = min(15, np.min(ys))                                                     # top left y margin
                    margin_y2 = 15 if (np.max(ys) + 15) < image_height else (image_height - np.max(ys)) # bottom right y margin

                    bbox_size_x = (bottom_right[0] - top_left[0] + margin_x1 + margin_x2)
                    bbox_size_y = (bottom_right[1] - top_left[1] + margin_y1 + margin_y2)

                    width = bbox_size_x/ image_width
                    height = bbox_size_y / image_height

                    x_center = ((top_left[0] - margin_x1) + bbox_size_x/2 ) / image_width
                    y_center = ((top_left[1] - margin_y1) + bbox_size_y/2 ) /image_width

                    print(top_left, bottom_right, margin_x1, margin_x2, margin_y1, margin_y2)

                    # Write the annotation to the YOLO file
                    f.write(f'{category_id} {round(x_center,6)} {round(y_center,6)} {round(width,6)} {round(height, 6)} ')

                    # Append normalized keypoints to the annotation
                    for i in range(0, len(keypoints), 3):
                        x = round(keypoints[i] / image_width, 6)
                        y = round(keypoints[i + 1] / image_height, 6)
                        v = round(keypoints[i + 2], 6)
                        f.write(f'{x} {y} {v} ')
                    f.write('\n')
                
                # keypointが格納されていない場合
                else:
                    # Write the annotation to the YOLO file  category, center, center, width, height
                    f.write(f'{category_id} {0} {0} {0} {0} ')

                    for i in range(POINT_NUM):
                        f.write(f'{0} {0} {0} ')


    print('Conversion complete.')


# Example usage
# coco_json_path = '/workspace/shizuka_labelbox/coco_format.json'
coco_json_path = '/workspace/shizuka_labelbox/coco_all_info_yobiset.json'
output_dir = '/workspace/data/ultrasound/dataset/yolo_all_frames'
convert_coco_to_yolo(coco_json_path, output_dir)