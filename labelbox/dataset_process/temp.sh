# 画像が保存されているディレクトリ
image_dir="/workspace/shizuka_keypointdetection/YOLOv7-POSE-on-Custom-Dataset/labelbox/data/images"

# ラベルを保存するディレクトリ
label_dir="/workspace/shizuka_keypointdetection/YOLOv7-POSE-on-Custom-Dataset/labelbox/data/labels"

# コピーするテンプレートファイル
template_file="${label_dir}/0000000.txt"

# .bmpファイルを検索し、それぞれに対してループ処理
find "${image_dir}" -type f -name "*.bmp" | while read bmp_file; do
  # ファイル名の拡張子を除いた部分を取得
  base_name=$(basename "${bmp_file}" .bmp)
  
  # 新しいラベルファイルのフルパスを生成
  new_label_file="${label_dir}/${base_name}.txt"
  
  # テンプレートファイルを新しいラベルファイルにコピー
  cp "${template_file}" "${new_label_file}"
  
  echo "Created ${new_label_file}"
done