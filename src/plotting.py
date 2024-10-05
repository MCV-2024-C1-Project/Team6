import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

class ImageNavigator:
    def __init__(self, results, feature_names):
        """
        Initialize the ImageNavigator with results and feature names.

        Parameters:
            results (list): A list of tuples (query_input, score_list, apk).
            feature_names (list): A list of feature names for plotting.
        """
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
        ax.set_title(f"Input Image: {os.path.basename(input_image_path)} {string_score}")
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
        for i, (sim_path, sim_image, sim_metric) in enumerate(zip(similar_paths, similar_images, similar_scores)):
            ax = self.fig.add_subplot(gs[i + 1, 0])
            self.axes[i + 1, 0] = ax
            ax.imshow(sim_image)
            ax.set_title(f"{i + 1}: {os.path.basename(sim_path)} | Distance: {sim_metric:.2f}")
            ax.axis('off')

            for col in range(1, num_cols):
                ax = self.fig.add_subplot(gs[i + 1, col])
                self.axes[i + 1, col] = ax
                x = np.arange(len(similar_features[i][col - 1]))
                ax.plot(x, similar_features[i][col - 1], color='orange')
                ax.fill_between(x, similar_features[i][col - 1], color='orange', alpha=0.3)
                ax.set_title(f"{self.feature_names[col - 1]} - Similar Image {i + 1}")
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