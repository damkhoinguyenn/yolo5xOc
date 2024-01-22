import matplotlib.pyplot as plt
import pandas as pd

# Load data from the text files
df1 = pd.read_csv('/runs/track/HandOc(0.5,0.5)/tracks/handAndStand.txt', delimiter=' ', header=None, names=['Frame', 'ID'])
df2 = pd.read_csv('C:/Users/dammi/Documents/StrongxYolo5/runs/track/HandStrong(0.5,0.5)/tracks/handAndStand.txt', delimiter=' ', header=None, names=['Frame', 'ID'])

# Convert the pandas Series to numpy arrays for plotting
frames1 = df1['Frame'].to_numpy()
ids1 = df1['ID'].to_numpy()
frames2 = df2['Frame'].to_numpy()
ids2 = df2['ID'].to_numpy()

# Create the plot with the desired figure size
fig, ax = plt.subplots(figsize=(10, 4))  # Adjust the size as needed

# Plot data from the first file
ax.plot(frames1, ids1, alpha=0.5, linewidth=2, color='blue', label='HandOc: conf_thres=0.5, iou_thres=0.5')

# Plot data from the second file
ax.plot(frames2, ids2, alpha=0.5, linewidth=2, color='red', label='HandStrong: conf_thres=0.5, iou_thres=0.5')

# Draw lines at x=0 and y=0
ax.axhline(y=0, color='k', linewidth=1)  # Horizontal line at y=0
ax.axvline(x=0, color='k', linewidth=1)  # Vertical line at x=0

# Set the limits for x and y axes
ax.set_xlim([min(frames1.min(), frames2.min()) - 1, max(frames1.max(), frames2.max()) + 1])  # Adjust as needed
ax.set_ylim([min(ids1.min(), ids2.min()) - 1, max(ids1.max(), ids2.max()) + 1])  # Adjust as needed

# Add legends, labels, etc.
ax.legend()
ax.set_xlabel('Frames')
ax.set_ylabel('IDs')

plt.tight_layout()

# Save or display the plot
save_img = input("Do you want to save the plot? (y/n): ").strip().lower()
if save_img == 'y':
    file_name = input("Enter the filename for the image: ").strip()
    file_path = f'{file_name}.jpg'  # Save as JPG
    fig.savefig(file_path, format='jpg', dpi=100)
    print(f"Plot saved as '{file_path}'.")
elif save_img == 'n':
    # If you don't want to save, just display the plot
    plt.show()
else:
    print("Invalid input. Image not saved.")
