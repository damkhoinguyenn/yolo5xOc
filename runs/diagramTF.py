import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator

def plot_frame_id_graph(file_path, output_file):
    # Load the data from the text file
    data = pd.read_csv(file_path, sep=' ', header=None, names=['frame', 'id'])
    data['id'] = data['id'].apply(lambda x: 0 if x == 0 else 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    def update(frame):
        ax.clear()
        left_limit = max(0, frame - 30)
        right_limit = max(frame, left_limit + 1)
        ax.set_xlim(left_limit, right_limit)

        ax.set_ylim(0, 1)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Detect (false or true)')
        ax.set_yticks([0, 1])

        # Set title with red background and double font size
        # ax.set_title('Standing Detection', backgroundcolor='red', color='white', fontsize=24)
        ax.set_title('Raising Hand Detection', backgroundcolor='blue', color='white', fontsize=24)

        ax.grid(False)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        for f, id in zip(data['frame'], data['id']):
            if id == 1 and f <= frame:
                ax.vlines(x=f, ymin=0, ymax=1, colors='red')

    # Rest of your code...

    # Creating the animation
    anim = FuncAnimation(fig, update, frames=range(1, data['frame'].max() + 1), interval=33)

    # Define progress callback function
    def progress_callback(current, total):
        print(f"Saving progress: {100 * (current / total):.2f}%", flush=True)

    # Save the animation
    print("Saving animation. This may take a while...")
    anim.save(output_file, writer='ffmpeg', progress_callback=progress_callback)
    print("Animation saved successfully.")

# Replace the file paths with your actual file paths
input_file_path = 'C:/Users/dammi/Documents/Web/data/std+hand_cam2_03_31_ca2/Hand.txt'
output_file_path = 'C:/Users/dammi/Documents/Web/data/std+hand_cam2_03_31_ca2/video3.mp4'

# Generate and save the animation
plot_frame_id_graph(input_file_path, output_file_path)
