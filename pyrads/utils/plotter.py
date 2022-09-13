#!/usr/bin/env python3
"""
Module with plotting functions
"""
# Standard libraries
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
    plt.show()
