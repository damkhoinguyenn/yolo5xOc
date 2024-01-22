import os

# đưa đường dẫn tuyệt đối
directory_path = 'C:/Users/dammi/Documents/OcxYolo5/yolov5/dataset/val/labels'

def is_valid_line(line):
    parts = line.split()
    if parts and parts[0] in ['0', '1']:
        return True
    return False

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        if not all(is_valid_line(line) for line in lines):
            os.remove(file_path)
            print(f"Deleted file: {filename}")

print("Done.")
