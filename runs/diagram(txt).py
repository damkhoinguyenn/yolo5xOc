import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load your data into a pandas DataFrame from a text file
df = pd.read_csv('C:/Users/dammi/Documents/diagramData/std-Hand-Oc.txt', delimiter=' ', header=None, names=['Frame', 'ID'])

# Create the plot with the desired figure size
fig, ax = plt.subplots(figsize=(12, 4))  # Adjusted figure size for better layout

# Group data by ID and plot each group with a high contrast color
colors = plt.cm.nipy_spectral(np.linspace(0, 1, df['ID'].nunique()))  # Using nipy_spectral colormap for high contrast
for (ID, group), color in zip(df.groupby('ID'), colors):
    # Convert pandas Series to numpy arrays before plotting
    frames = group['Frame'].to_numpy()
    ids = group['ID'].to_numpy()
    ax.plot(frames, ids, alpha=1, linewidth=4, color=color)  # Increased alpha and line width for stronger visibility

ax.axhline(y=0, color='k', linewidth=1)
ax.axvline(x=0, color='k', linewidth=1)

ax.set_xlim([df['Frame'].min() - 1, df['Frame'].max() + 1])
ax.set_ylim([df['ID'].min() - 1, df['ID'].max() + 1])

ax.set_xlabel('Frames')
ax.set_ylabel('IDs')

plt.tight_layout()  # Should work better without the legend

# Save or display the plot
save_img = input("Do you want to save the plot? (y/n): ").strip().lower()
if save_img == 'y':
    file_name = input("Enter the filename for the image: ").strip()
    file_path = f'{file_name}.jpg'
    fig.savefig(file_path, format='jpg', dpi=100)
    print(f"Plot saved as '{file_path}'.")
elif save_img == 'n':
    plt.show()
else:
    print("Invalid input. Image not saved.")
