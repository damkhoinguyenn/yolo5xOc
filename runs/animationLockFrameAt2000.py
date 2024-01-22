import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator

# Load your data into a pandas DataFrame
df = pd.read_csv('/runs/track/HandOc(0.5,0.5)/tracks/handAndStand.txt', delimiter=' ',
                 header=None, names=['Frame', 'ID'])

# Convert both 'Frame' and 'ID' columns to integers
df['Frame'] = df['Frame'].astype(int)
df['ID'] = df['ID'].astype(int)

# Define total_data_points
total_data_points = len(df)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 4))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel('Frame')
ax.set_ylabel('ID')

# Create a line plot object for the animation
line, = ax.plot([], [], linestyle='-', color='red')

# Number of frames to display at any time
max_display_frames = 3000

# Animation function
def animate(i):
    if i >= total_data_points:
        return line,

    frame_start_number = max(0, df['Frame'].iloc[i] - max_display_frames)
    start_index = df[df['Frame'] >= frame_start_number].index[0]
    end_index = i + 1

    visible_df = df.iloc[start_index:end_index]
    line.set_data(visible_df['Frame'], visible_df['ID'])

    if not visible_df.empty:
        min_frame, max_frame = visible_df['Frame'].min(), visible_df['Frame'].max()
        min_id, max_id = visible_df['ID'].min(), visible_df['ID'].max()

        # Add a buffer to axis limits if min and max are the same
        frame_buffer = 1 if min_frame == max_frame else 0
        id_buffer = 0.5 if min_id == max_id else 0

        ax.set_xlim(min_frame - frame_buffer, max_frame + frame_buffer)
        ax.set_ylim(min_id - id_buffer, max_id + id_buffer)

    return line,

# Initialize the animation
def init():
    line.set_data([], [])
    return line,

# Call the animator
anim = FuncAnimation(fig, animate, init_func=init, frames=len(df), interval=100, blit=False)

# Ask user for the output file name
output_filename = input("Enter the name of the animation file: ")
if not output_filename.endswith('.mp4'):
    output_filename += '.mp4'

def progress_callback(current, total):
    progress = 100 * (current / total)
    print(f"Saving progress: {progress:.2f}%", flush=True)

# Save the animation
anim.save(output_filename, writer='ffmpeg', fps=30, progress_callback=lambda current, total: print(f"Saving progress: {100 * (current / total):.2f}%"))

plt.close(fig)
