# yolo5xOc
 Yolov5(v6.2) with OcSort
# env:
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
# run detect:
python track.py --source handAndStand.mp4 --yolo-weights weights/bestStand.pt --save-txt --tracking-method ocsort
