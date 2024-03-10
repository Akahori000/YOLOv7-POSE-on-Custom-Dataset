import pydicom
import cv2
import imageio
import os
import numpy as np
from PIL import Image
import pandas as pd


def frames_to_video(frames, output_path, fps):
    # imageioを使ってMP4ビデオに書き込む
    with imageio.get_writer(output_path, format='FFMPEG', fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

def dicom_to_bmp_mp4(dicom_folder, output_bmp_path, output_mp4_path, left, top, right, bottom, crop, fps=30):
    # DICOMフォルダ内のすべてのDICOMファイルを取得
    dicom_files = [os.path.join(dicom_folder, file) for file in os.listdir(dicom_folder)]# if file.endswith('.dcm')]
    # dicom_files = [os.path.join(dicom_folder, file) for file in os.listdir(dicom_folder)]
    # DICOMファイルを読み込んで画像をリストに保存
    
    # フォルダにあるすべてのファイルに対して実行
    for file_path in dicom_files:
        ds = pydicom.dcmread(file_path)
        pixel_array = ds.pixel_array

        images = []
        for frame_num in range(pixel_array.shape[0]):
            # OpenCVで画像をRGB形式に変換
            im = cv2.cvtColor(pixel_array[frame_num], cv2.COLOR_YUV2RGB) 
            image = Image.fromarray(im) 
            if crop == True:
                image = image.crop((left, top, right, bottom))
            image = np.array(image)

            images.append(image)
    
        frames = np.array(images)


        filename = os.path.basename(file_path)
        f_name = os.path.splitext(filename)[0]

        # 画像をbmpで保存
        if not os.path.exists(output_bmp_path):
            os.makedirs(output_bmp_path)

        for i in range (len(frames)):
            image = Image.fromarray(frames[i])
            image.save(f"{output_bmp_path}/{f_name}_{str(i+1)}.bmp")

        # 画像をMP4ビデオに変換
        if not os.path.exists(output_mp4_path):
            os.makedirs(output_mp4_path)
        dst_path = f"{output_mp4_path}/{f_name}.mp4"

        frames_to_video(frames, dst_path, ds.CineRate)
        print("dicom:", file_path, "frame_num:" , len(images))

# こういう感じ
  # annotated_good =  ['37', '55', '78', '104'] # 開始フレーム、終了フレームの対の繰り返し
  # annotated_perfect = ['1', '36', '56', '77', '105', '183']
def _is_labeled(frame, list):
    pair_num = int(len(list)/2)
    flg = False
    for i in range(pair_num):
        if int(list[i*2]) <= frame <= int(list[(i*2)+1]):
            flg = True
            return flg
    return flg

def output_labels(dicom_folder, my_database):
   # DICOMフォルダ内のすべてのDICOMファイルを取得
    dicom_files = [os.path.join(dicom_folder, file) for file in os.listdir(dicom_folder) if file.endswith('.dcm')]
    
    # フォルダにあるすべてのファイルに対して実行
    for file_path in dicom_files:
        ds = pydicom.dcmread(file_path)
        pixel_array = ds.pixel_array

        filename = os.path.basename(file_path)
        f_name = os.path.splitext(filename)[0]

        # キーが4の場合の値を取り出す
        values = next((item["values"] for item in my_database if item["key"] == f_name), None)
        if values is not None:
            # print(values) 
            print(f_name)
            print("perfect", values[0])
            print("good", values[1])

        else:
            print("キーが存在しません", f_name)
    
        annotation = []
        for frm in range(len(pixel_array)):
            # if frm == 177:
                # print("koko")
            label = "no_label"
            frame_num = frm + 1 # LabelBoxが1からカウントしている
            if _is_labeled(frame_num, values[1]) == True:
                print(frame_num, "good")
                label = "Good"
                annotation.append(1)
            elif _is_labeled(frame_num, values[0]) == True:
                print(frame_num, "perfect")
                label = "Perfect"
                annotation.append(2)
            else:
                print(frame_num, "no_label") 
                label = "No_label"    
                annotation.append(0)  
        labels = np.array(annotation)
        np.savetxt(f"{dicom_folder}/label/{f_name}_label.csv", labels, fmt="%d", delimiter=",")


def trim_and_save_to_each_folder(dicom_folder, label_path, output_bmp_path, left, top, right, bottom , fps=30):
    # DICOMフォルダ内のすべてのDICOMファイルを取得
    dicom_files = [os.path.join(dicom_folder, file) for file in os.listdir(dicom_folder) if file.endswith('.dcm')]
    
    # フォルダにあるすべてのファイルに対して実行
    for file_path in dicom_files:
        ds = pydicom.dcmread(file_path)
        pixel_array = ds.pixel_array


        filename = os.path.basename(file_path)
        f_name = os.path.splitext(filename)[0]

        #labelを読み込む
        df = pd.read_csv(f"{label_path}/{f_name}_label.csv", header=None, index_col=None) # 0からよみこんでしまう
        df = np.array(df)



        #読み込んだlabelに従って、dicomから読み込んだbmpをgood/perfect/othersフォルダに保存
        for frame_num in range(pixel_array.shape[0]): # こっちも0から読み込む
            im = cv2.cvtColor(pixel_array[frame_num], cv2.COLOR_YUV2RGB) 
            image = Image.fromarray(im) 
            image = image.crop((left, top, right, bottom))
            if df[frame_num] == 1:
                save_path = f"{output_bmp_path}/good/"
            elif df[frame_num] == 2:
                save_path = f"{output_bmp_path}/perfect/"
            else:
                save_path = f"{output_bmp_path}/others/"

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            image.save(f"{save_path}/{f_name}_{str(frame_num+1)}.bmp")


        
    

# DICOMフォルダのパス出力BMP, MP4ファイルのパスを指定して変換を実行
dicom_folder_path = '/workspace/data/ultrasound/dataset240118/DICOM'
output_video_path = '/workspace/data/ultrasound/dataset240118/mp4'
output_bmp_path = '/workspace/data/ultrasound/dataset240118/bmp'

output_video_path_no_crop = '/workspace/data/ultrasound/dataset240118/no_crop_mp4'
output_bmp_path_no_crop = '/workspace/data/ultrasound/dataset240118/no_crop_bmp'


label_path = f"{dicom_folder_path}/label"

left, top, right, bottom = [141, 74, 708, 599]

# dicomフォルダを読み込んで/bmpに保存
dicom_to_bmp_mp4(dicom_folder_path, output_bmp_path_no_crop, output_video_path_no_crop, left=left, top=top, right=right, bottom=bottom, crop=False)
dicom_to_bmp_mp4(dicom_folder_path, output_bmp_path, output_video_path, left=left, top=top, right=right, bottom=bottom, crop=True)

my_database_teruya = [              # perfect  # good
    {"key": "00000004", "values": [[180, 194], [51, 56, 176, 179]]},
    {"key": "00000005", "values": [[176, 184], [164, 175]]},
    {"key": "00000006", "values": [[253, 257], [84, 93, 244, 252, 258, 292]]},
    {"key": "00000010", "values": [[], [129, 141]]},
    {"key": "00000011", "values": [[57, 77], [1, 56, 107, 110, 167, 211]]},
    {"key": "00000012", "values": [[], [44,45, 62, 75, 108,183]]},
    {"key": "00000015", "values": [[], [108,129, 178, 194]]},
    {"key": "00000018", "values": [[], [95,131,204,210,312,355]]}
]
my_database_michishige = [
    {"key": "00000004", "values": [[45, 48], [1, 6, 49, 63, 166, 191]]},
    {"key": "00000005", "values": [[], [197, 215]]},
    {"key": "00000006", "values": [[242, 262], [48, 107, 182, 241, 263, 292]]},
    {"key": "00000010", "values": [[], [89, 147]]},
    {"key": "00000011", "values": [[38, 83, 154, 211], [1, 37, 89, 118, 136, 153]]},
    {"key": "00000012", "values": [[52, 63, 139, 183], [1,51,64, 138]]},
    {"key": "00000015", "values": [[97, 133, 200, 206], [53, 61, 134, 140, 174, 199, 311, 365]]},
    {"key": "00000018", "values": [[192, 243, 311, 366], [244, 244, 306, 310, 390, 460]]}
]

my_database_tsuge = [
    {"key": "00000004", "values": [[177,190 ], [49,59,170,176]]},
    {"key": "00000005", "values": [[177,186], [155, 176]]},
    {"key": "00000006", "values": [[248, 263, 280, 292], [242,247,264,279]]},
    {"key": "00000010", "values": [[125,135], [111,124,136,140]]},
    {"key": "00000011", "values": [[26,31,183,193], [1,25,32,51,100,108,163,182]]},
    {"key": "00000012", "values": [[1,9,156,183], [10,22,65,74,108,155]]},
    {"key": "00000015", "values": [[], [121,128]]},
    {"key": "00000018", "values": [[94,98], [81,93,99,118,201,232]]}
]

# dicomフォルダと上記dicより、dicom/labelにlabelを保存
# output_labels(dicom_folder_path, my_database_michishige)


# dicomフォルダから読み込んだbmpをperfect/good/no_labelに保存
# trim_and_save_to_each_folder(dicom_folder_path, label_path, output_bmp_path, left=left, top=top, right=right, bottom=bottom)

