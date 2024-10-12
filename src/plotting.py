import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import re
import matplotlib.animation as animation
def image_name_to_id(image_name):
    match = re.search(r'(\d+)', image_name)
    if match:
        return int(match.group(1))
    return -1

class ImageNavigator:
    def __init__(self, results, feature_names):
        self.results = results  # List of (query_input, score_list, apk)
        self.feature_names = feature_names
        self.index = 0  # Current index in the results list
        self.total = len(results)
        self.fig = plt.figure(figsize=(12, 8),constrained_layout=True)
        self.fig.canvas.manager.set_window_title('Image Navigator')
        self.axes = None
        # Connect the key press event handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self._update_plot()

    def _update_plot(self):
        # Clear the figure but keep the window open
        self.fig.clear()
        query_input, score_list, apk = self.results[self.index]
        input_image_path = query_input[0]
        input_image = Image.open(input_image_path)
        input_feature = query_input[1]

        similar_paths = [f[0] for f in score_list]
        similar_images = [Image.open(f[0]) for f in score_list]
        similar_scores = [f[1] for f in score_list]
        similar_raw_scores = [f[3] for f in score_list]
        similar_features = [f[2] for f in score_list]
        string_score = f"| Performance: {apk:.2f}" if apk is not None else ""

        num_rows = len(score_list) + 1
        num_cols = 1 + len(input_feature)

        # Create a GridSpec for layout
        gs = self.fig.add_gridspec(num_rows, num_cols, top=0.9, bottom=0.1)
        self.axes = np.empty((num_rows, num_cols), dtype=object)

        # Plot the input image
        ax = self.fig.add_subplot(gs[0, 0])
        self.axes[0, 0] = ax
        ax.imshow(input_image)
        ax.set_title(f"{image_name_to_id(os.path.basename(input_image_path))} {string_score}")
        ax.axis('off')

        # Plot features of the input image
        for col in range(1, num_cols):
            ax = self.fig.add_subplot(gs[0, col])
            self.axes[0, col] = ax
            x = np.arange(len(input_feature[col - 1]))
            ax.plot(x, input_feature[col - 1], color='blue')
            ax.fill_between(x, input_feature[col - 1], color='blue', alpha=0.3)
            ax.set_title(f"{self.feature_names[col - 1]} - Input Image")
            ax.set_xticks([])
            ax.set_yticks([])

        # Plot similar images and their features
        for i, (sim_path, sim_image, sim_metric, raw_sim_metric) in enumerate(zip(similar_paths, similar_images, similar_scores, similar_raw_scores)):
            ax = self.fig.add_subplot(gs[i + 1, 0])
            self.axes[i + 1, 0] = ax
            ax.imshow(sim_image)
            ax.set_title(f"{i + 1}: {image_name_to_id(os.path.basename(sim_path))} | Distance: {sim_metric:.2f}")
            ax.axis('off')

            for col in range(1, num_cols):
                ax = self.fig.add_subplot(gs[i + 1, col])
                self.axes[i + 1, col] = ax
                x = np.arange(len(similar_features[i][col - 1]))
                ax.plot(x, similar_features[i][col - 1], color='orange')
                ax.fill_between(x, similar_features[i][col - 1], color='orange', alpha=0.3)
                ax.set_title(f"{self.feature_names[col - 1]} - Distance: {raw_sim_metric[col-1]:.2f}")
                ax.set_xticks([])
                ax.set_yticks([])

        self.fig.suptitle('-'.join(self.feature_names) + f" ({self.index + 1}/{self.total})")
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        if event.key == 'right':
            # Move to the next image
            if self.index < self.total - 1:
                self.index += 1
                self._update_plot()
        elif event.key == 'left':
            # Move to the previous image
            if self.index > 0:
                self.index -= 1
                self._update_plot()
        elif event.key == 'escape':
            plt.close(self.fig)

    def show(self):
        plt.show()


def show_image3d(image, axis='axial', aspect=1, vmin=None, vmax=None, cmap='magma', max_slices=100, mask=None, target=None, figsize=(6,6), save=False, show=True):
    if axis=='axial':
        frames = lambda i: image[i]
        mask_frames = lambda i: mask[i]
        target_frames = lambda i: target[i]
        ns = image.shape[0]
        origin='upper'
    elif axis=='sagittal':
        frames = lambda i: image[:,:,i]
        mask_frames = lambda i: mask[:,:,i]
        target_frames = lambda i: target[:,:,i]
        ns = image.shape[2]
        origin='lower'
    elif axis=='coronal':
        frames = lambda i: image[:,i,:]
        mask_frames = lambda i: mask[:,i,:]
        target_frames = lambda i: target[:,i,:]
        ns = image.shape[1]
        origin='lower'
    else:
        raise ValueError(f'Invalid axis{axis}!')

    nf = min(max_slices, ns)
    x = [int(i*ns / nf) for i in range(nf)]
    fig, ax = plt.subplots(figsize=figsize)
    ims = []
    for i in x:
        artist = []
        img = frames(i)
        im = ax.imshow(img, animated=True, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest", origin=origin)
        artist.append(im)
        ax.set_aspect(aspect)
        txt = ax.text(0.5, 1.05, f'Slice: {i}', horizontalalignment='center', fontsize='large', transform=ax.transAxes)
        artist.append(txt)
        if target is not None:
            tf= target_frames(i)
            target_image = np.zeros((img.shape[0], img.shape[1], 4))
            target_image[:,:, 3] = tf*0.5
            target_image[:,:, 0] = 0.1
            target_image[:,:, 1] = 0.8
            target_image[:,:, 2] = 0.1
            im2 = ax.imshow(target_image, animated=True, interpolation="nearest", origin=origin)
            artist.append(im2)
        if mask is not None:
            mf = mask_frames(i)
            mask_image = np.zeros((img.shape[0], img.shape[1], 4))
            mask_image[:,:, 3] = mf*0.5
            mask_image[:,:, 0] = 0.8
            mask_image[:,:, 1] = 0.1
            mask_image[:,:, 2] = 0.1
            im3 = ax.imshow(mask_image, animated=True, interpolation="nearest", origin=origin)
            artist.append(im3)
        ims.append(artist)
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
    plt.show()
    return ani