import os

# đưa đường dẫn tuyệt đối
directory_path = 'C:/Users/dammi/Documents/OcxYolo5/yolov5/dataset/train/labels'

def replace_first_number(line):
    parts = line.split()
    if parts[0] == '4':
        parts[0] = '0'
    elif parts[0] == '2':
        parts[0] = '1'
    return ' '.join(parts) + '\n'

for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        new_lines = [replace_first_number(line) for line in lines]

        with open(file_path, 'w') as file:
            file.writelines(new_lines)

print("Done.")
