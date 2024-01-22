import pandas as pd

def fill_missing_frames(input_file_path, output_file_path):
    # Load the data from the text file
    data = pd.read_csv(input_file_path, sep=' ', header=None, names=['frame', 'id'])

    # Find the maximum frame number
    max_frame = data['frame'].max()

    # Create a DataFrame with all frames from 1 to max_frame
    all_frames = pd.DataFrame({'frame': range(1, max_frame + 1)})
    all_frames['id'] = 0  # Initialize all ids to 0

    # Update the id for frames present in the original data
    for index, row in data.iterrows():
        all_frames.loc[all_frames['frame'] == row['frame'], 'id'] = row['id']

    # Save the modified data to a new file
    all_frames.to_csv(output_file_path, sep=' ', header=False, index=False)

# Replace the file paths with your actual file paths
input_file_path = 'C:/Users/dammi/Documents/Web/data/std+hand_cam1_03_31_ca2/Hand.txt'
output_file_path = 'C:/Users/dammi/Documents/Web/data/std+hand_cam1_03_31_ca2/Hand2.txt'

# Process the file
fill_missing_frames(input_file_path, output_file_path)
