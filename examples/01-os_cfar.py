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
import pyrads.algms.fft
import pyrads.algms.os_cfar
import pyrads.algms.remove_offset
import pyrads.algms.window
import pyrads.pipeline
import pyrads.pipes.preproc_pipeline
import pyrads.utils.plotter


def main():
    # User-defined parameters for the pipeline
    dataset_params = {
        "dataset": "raw_data/scenario1_0",
    }
    h5_handler = dhandler.h5_handler.H5Handler()
    data, radar_config, calib_vec = h5_handler.load(
        dataset_params["dataset"],
        dataset_dir=None
    )
    radar_dataframes = data[60]

    # Define algorithms parameters
    window_params = {
        "axis": -1,
        "window_type": "hann"
    }
    fft_params = {
        "type": "range",
    }
    oscfar_params = {
        "n_dims": 1,
        "window_width": 16,
        "ordered_k": 6,
        "alpha": 0.2,
        "n_guard_cells": 2,
    }
    remove_offset_alg = pyrads.algms.remove_offset.RemoveOffset(
        in_data_shape=radar_dataframes.shape
    )
    window_alg = pyrads.algms.window.Window(
        in_data_shape=remove_offset_alg.out_data_shape,
        **window_params
    )
    range_fft_alg = pyrads.algms.fft.FFT(
        in_data_shape=window_alg.out_data_shape,
        **fft_params
    )
    oscfar_alg = pyrads.algms.os_cfar.OSCFAR(
        in_data_shape=range_fft_alg.out_data_shape,
        **oscfar_params
    )

    # Create Pipeline instance with the list of defined algorithms
    algorithms = [
        remove_offset_alg,
        window_alg,
        range_fft_alg,
        oscfar_alg]
    pipeline = pyrads.pipeline.Pipeline(algorithms)
    pipe_data = pipeline(radar_dataframes)
    fft_out = pipe_data[-2]
    out = pipe_data[-1]
    
    # Plot results
    fft_data = fft_out[0,0,25,:]
    out_data = out[0,0,25,:]
    pyrads.utils.plotter.plot_single_ramp_pipeline(fft_data, out_data)
    return


if __name__ == "__main__":
    main()
