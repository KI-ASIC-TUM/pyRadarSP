#!/usr/bin/env python3
"""
Module with plotting functions
"""
# Standard libraries
import imageio
import matplotlib.pyplot as plt
import numpy as np


def plot_single_ramp_pipeline(
        image,
        in_data,
        out,
        in_title="FFT",
        out_title="OS-CFAR"
    ):
    """
    Plot input and output data of a single ramp of a 1D pipeline
    """
    fig, axs = plt.subplots(3, figsize=(10,12))
    axs[0].imshow(image)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].plot(in_data)
    axs[1].set_title(in_title)
    axs[2].plot(out)
    axs[2].set_title(out_title)
    fig.tight_layout()
    fig.savefig("single_chirp.eps", dpi=800)
    plt.show()


def plot_multi_ramp_pipeline(
        images,
        in_data,
        out_data,
        in_title="FFT",
        out_title="OS-CFAR"):
    fig, ax = plt.subplots(3, figsize=(10,12))
    plotter = ScrollPlotter(fig, ax, images, in_data, out_data, in_title, out_title)
    plotter.sequence2gif()
    fig.canvas.mpl_connect('scroll_event', plotter.on_scroll)
    fig.tight_layout()
    plt.show()


def plot_rd_map(image, fft_data):
    """
    Plot the range-Doppler map together witht the scene image
    """
    n_ramps, n_samples = fft_data.shape
    doppler_bins = int(n_ramps/2)
    fig, ax = plt.subplots(2, figsize=(10,12))
    ax[0].imshow(image)
    ax[0].set_xlabel("Range (bins)")
    ax[0].set_ylabel("Velocity (bins)")
    ax[1].imshow(fft_data, extent=[0, n_samples,-(doppler_bins-1),doppler_bins])
    fig.tight_layout()
    fig.savefig("rd_map.eps", dpi=800)
    plt.show()


class ScrollPlotter():
    def __init__(self, fig, ax, images, in_data, out_data, in_title, out_title):
        # Data to be plotted
        self.images = images
        self.in_data = in_data
        self.out_data = out_data
        # Plotter metadata
        self.fig = fig
        self.ax = ax
        self.slices = self.images.shape[0]
        self.ind = 0
        self.init_plot(in_title, out_title)

    def init_plot(self, in_title, out_title):
        """
        Set the initial data to be shown in the plots
        """
        self.im = self.ax[0].imshow(self.images[self.ind])
        self.ax[0].set_xticks([])
        self.ax[0].set_yticks([])
        self.ax[0].set_title("Frame {}".format(self.ind))
        self.line1, = self.ax[1].plot(self.in_data[self.ind])
        self.ax[1].set_title(in_title)
        self.line2, = self.ax[2].plot(self.out_data[self.ind])
        self.ax[2].set_title(out_title)
        return

    def on_scroll(self, event):
        """
        Reaction to the scroll event in the mouse
        """
        if event.button == 'up':
            self.ind = (self.ind+1) % self.slices
        else:
            self.ind = (self.ind-1) % self.slices
        self.update()
        return

    def update(self):
        """
        Update the content displayed in the plots
        """
        # Update data on the three subplots
        self.im.set_data(self.images[self.ind])
        self.line1.set_ydata(self.in_data[self.ind])
        self.line2.set_ydata(self.out_data[self.ind])
        # Indicate current frame in the title
        self.ax[0].set_title("Frame {}".format(self.ind))
        # Refresh the drawing of the plot
        self.im.axes.figure.canvas.draw()
        return

    def sequence2gif(self, fps=5):
        """
        Create a gif animation with the sequence of samples

        Iterate over the time steps and generate a gif with the three
        subplots generated in the plotter.

        Corrupted frames are ignored. TODO: Remove corrupted frames from
        the dataset.
        """
        sequence = []
        for i in range(17, 59):
            # Ignore corrupted frames
            # Scene 01:
            if i in [19, 27, 31, 35, 39, 42, 44, 47,49, 58]:
            # Scene 04:
            # if i in [7, 11, 20, 22, 24, 30, 32, 38,50, 52, 59, 62, 65, 69, 83,
            #          87, 90, 94, 100, 103, 108, 113, 115, 121, 125, 129, 131]:
                continue
            self.ind = i
            self.update()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            im_shape = self.fig.canvas.get_width_height()[::-1] + (3,)
            image  = image.reshape(im_shape)
            sequence.append(image)
        self.ind = 0
        imageio.mimsave('./sequence.gif', sequence, fps=fps)
        pass
