#!/usr/bin/env python3
"""
Run OS-CFAR over test dataset
"""
# Standard libraries
import numpy as np
# Local libraries
import dhandler
import dhandler.h5_handler
import dhandler.signal_processing
import dhandler.database
import pyrads.algms.identity
import pyrads.algms.os_cfar
import pyrads.pipeline
import pyrads.pipes.preproc_pipeline
import pyrads.utils.plotter


def load_data(**kwargs):
    h5_handler = dhandler.h5_handler.H5Handler()
    data, radar_config, calib_vec = h5_handler.load(kwargs["dataset"], dataset_dir=None)
    radar_dataframes = data
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
    return np.array(range_fft_frames)


def main():
    # User-defined parameters for the pipeline
    dataset_params = {
        "dataset": "raw_data/scenario1_0",
    }
    data = load_data(**dataset_params)[20:25]
    # User-defined parameters for os-cfar algorithm
    oscfar_params = {
        "in_data_shape": data.shape,
        "n_dims": 1,
        "window_width": 16,
        "ordered_k": 3,
        "alpha": 0.2,
        "n_guard_cells": 2,
    }
    # Create a pipeline instance and add the data and a signal processing
    # algorithm to it.
    oscfar_alg = pyrads.algms.os_cfar.OSCFAR(**oscfar_params)
    id_alg = pyrads.algms.identity.Identity(in_data_shape=oscfar_alg.out_data_shape)

    algorithms = [oscfar_alg, id_alg]
    pipeline = pyrads.pipeline.Pipeline(algorithms)
    out = pipeline(data)[-1]
    
    # Plot results
    pyrads.utils.plotter.plot_single_ramp_pipeline(np.abs(data[0]), out[0])
    return


if __name__ == "__main__":
    main()
