import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

# Load your data into a pandas DataFrame
df = pd.read_csv('/runs/track/StandOc(0.5,0.5)/tracks/handAndStand.txt', delimiter=' ',
                 header=None, names=['Frame', 'ID'])

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Extract data
x_data, y_data = df['Frame'], df['ID']

# Create a line plot object (no markers)
line, = ax.plot([], [], linestyle='-', marker='')

# Set the initial view limits
ax.set_xlim((x_data.min(), x_data.max()))
ax.set_ylim((y_data.min(), y_data.max()))


# Initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,


# Animation function which updates figure data. This is called sequentially
def animate(i):
    # Update the line plot with all points up to the current frame
    line.set_data(x_data[:i + 1], y_data[:i + 1])
    return line,


# Progress callback function
def progress_callback(current, total):
    progress = 100 * (current / total)
    print(f"Saving progress: {progress:.2f}%")


# Call the animator
anim = FuncAnimation(fig, animate, init_func=init, frames=len(x_data), interval=20, blit=True)

# Ask user for the output file name
output_filename = input("Enter the name of the animation file: ")
if not output_filename.endswith('.mp4'):
    output_filename += '.mp4'

# Save the animation with progress callback
anim.save(output_filename, writer='ffmpeg', fps=30, progress_callback=progress_callback)

plt.close(fig)  # Close the figure when done
