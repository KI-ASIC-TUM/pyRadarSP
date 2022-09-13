#!/usr/bin/env python3
"""
Module with plotting functions
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np


def plot_single_ramp_pipeline(
        in_data,
        out,
        in_title="FFT",
        out_title="OS-CFAR"
    ):
    """
    Plot input and output data of a single ramp of a 1D pipeline
    """
    fig, axs = plt.subplots(2)
    axs[0].plot(in_data)
    axs[0].set_title(in_title)
    axs[1].plot(out)
    axs[1].set_title(out_title)
    fig.tight_layout()
    plt.show()
