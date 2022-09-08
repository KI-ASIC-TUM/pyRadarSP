#!/usr/bin/env python3
"""
Run OS-CFAR over test dataset
"""
# Standard libraries
import numpy as np
# Local libraries
import dhandler
import dhandler.hires
import dhandler.signal_processing
import dhandler.database
import pyrads.algms.os_cfar
import pyrads.pipeline
import pyrads.pipes.preproc_pipeline
import pyrads.utils.plotter


def load_data():
    radar_dataframes, radar_config = dhandler.hires.load_dataset('raw_data/scenario1_0', include_config=True)
    calib_vec = dhandler.hires.load_calibration('raw_data/scenario1_0', radar_config)


    # We do not care about the actual frames
    radar_dataframes = np.vstack(radar_dataframes)

    # Could be part of a chain link class
    time_range = dhandler.signal_processing.time_range(radar_config)
    distance_range = dhandler.signal_processing.distance_range(radar_config)

    # Save frames here
    range_fft_frames = []

    for frame, data in enumerate(radar_dataframes):
        # Pipeline
        no_offset_data = dhandler.signal_processing.remove_offset(data)
        smoothed_data = dhandler.signal_processing.smoothen(no_offset_data)
        range_fft_data = dhandler.signal_processing.range_fft(smoothed_data)

        for i in range(0, 32, 4):
            range_fft_frames.append(range_fft_data[2,0,i,:])
    return range_fft_frames


def main():
    # User-defined parameters for the pipeline
    dataset_params = {
        "dataset": "test_dataset_1",
    }
    # User-defined parameters for os-cfar algorithm
    preproc_params = {
        "smooth_k": 4,
        "fft_size": 256
    }
    oscfar_params = {
        "in_data_shape": (256,),
        "n_dims": 1,
        "window_width": 16,
        "ordered_k": 3,
        "alpha": 0.2,
        "n_guard_cells": 2,
    }
    data = load_data()[20]
    # TODO
    if False:
        data = pyrads.utils.data_loader.load_dataset(**dataset_params)
    # Create a pipeline instance and add the data and a signal processing
    # algorithm to it.
    if False:
        preproc_pipe = pyrads.pipes.preprocesing_pipeline.Pipeline(**preproc_params)
    oscfar_alg = pyrads.algms.os_cfar.OSCFAR(**oscfar_params)

    algorithms = [oscfar_alg]
    pipeline = pyrads.pipeline.Pipeline(algorithms)
    out = pipeline(data)

    #TODO: Plotting function in additional library
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)
    axs[0].plot(data)
    axs[0].set_title("FFT")
    axs[1].plot(out)
    axs[1].set_title("OS-CFAR")
    plt.tight_layout()
    plt.show()
    # Plot data before executing pipeline
    return


if __name__ == "__main__":
    main()
