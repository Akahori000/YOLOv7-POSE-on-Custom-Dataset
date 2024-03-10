import os
import shutil

def move_files_with_mask(source_folder, destination_folder):
    # フォルダ内のファイルを取得
    files = os.listdir(source_folder)

    for file_name in files:
        source_path = os.path.join(source_folder, file_name)

        # ファイル名に "mask" が含まれている場合は別のフォルダに移動
        if "mask" in file_name:
            destination_path = os.path.join(destination_folder, file_name)
            shutil.move(source_path, destination_path)
            print(f"File {file_name} moved to {destination_folder}.")

# ソースフォルダとデスティネーションフォルダのパスを指定
source_folder = "/workspace/data/echo/Dataset_BUSI_with_GT/normal"
destination_folder = "/workspace/data/echo/Dataset_BUSI_with_GT/normal_mask"
os.makedirs(destination_folder, exist_ok=True)

# 関数を呼び出して実行
move_files_with_mask(source_folder, destination_folder)