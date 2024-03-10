import cv2
import os
import pandas as pd
import numpy as np

def convert_mp4_to_bmp(input_folder, output_folder, label_path):
    # Output folderを作成
    os.makedirs(output_folder, exist_ok=True)

    # input_folder内の全mp4ファイルに対して処理
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_path = os.path.join(input_folder, filename)

            # VideoCaptureを作成
            cap = cv2.VideoCapture(input_path)

            # フレーム数を取得
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            filename = os.path.basename(input_path)
            f_name = os.path.splitext(filename)[0]

            #labelを読み込む
            df = pd.read_csv(f"{label_path}/{f_name}_label.csv", header=None, index_col=None) # 0からよみこんでしまう
            df = np.array(df)

            # フレームごとに処理
            for frame_num in range(total_frames):
                # フレームを読み込む
                ret, frame = cap.read()
                if not ret:
                    break

                # ファイル名を構築

                if df[frame_num] == 1:
                    save_path = f"{output_folder}/good/"
                elif df[frame_num] == 2:
                    save_path = f"{output_folder}/perfect/"
                else:
                    save_path = f"{output_folder}/others/"

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                output_filename = f"{os.path.splitext(filename)[0]}_frame_{frame_num:04d}.bmp"
                output_path = os.path.join(save_path, output_filename)

                # BMP形式で保存
                cv2.imwrite(output_path, frame)

            # VideoCaptureを解放
            cap.release()

# Example usage:
input_folder_path = '/workspace/data/ultrasound/dataset/splitted/mp4/test'
output_folder_path = '/workspace/data/ultrasound/dataset/splitted/bmp/test1'
label_path = "/workspace/data/ultrasound/dataset/dicom/label_teruya0"


convert_mp4_to_bmp(input_folder_path, output_folder_path, label_path)