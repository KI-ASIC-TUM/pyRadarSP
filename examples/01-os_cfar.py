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


def main(frame_n=30, chirp_n=30, multi_ramp=True):
    # User-defined parameters for the pipeline
    dataset_params = {
        "dataset": "raw_data/scenario1_0",
    }
    h5_handler = dhandler.h5_handler.H5Handler()
    data, radar_config, calib_vec = h5_handler.load(
        dataset_params["dataset"],
        dataset_dir=None
    )
    images = h5_handler.load_images(dataset_params["dataset"])
    images = np.array(images)
    image = images[frame_n]
    # Use information only from first Rx and first Tx antenna.
    # Keep same dimensionality.
    reduced_data = data[:,0,0,:,:]
    reduced_data = reduced_data.reshape(
            data.shape[0],
            1,
            1,
            data.shape[3],
            data.shape[4]
    )

    # Define algorithms parameters
    window_params = {
        "axis": -1,
        "window_type": "hann"
    }
    fft_params = {
        "type": "range",
        "normalize": True,
        "out_format": "modulus"
    }
    oscfar_params = {
        "n_dims": 1,
        "window_width": 16,
        "ordered_k": 6,
        "alpha": 0.2,
        "n_guard_cells": 2,
    }
    remove_offset_alg = pyrads.algms.remove_offset.RemoveOffset(
        reduced_data.shape
    )
    window_alg = pyrads.algms.window.Window(
        remove_offset_alg.out_data_shape,
        **window_params
    )
    range_fft_alg = pyrads.algms.fft.FFT(
        window_alg.out_data_shape,
        **fft_params
    )
    oscfar_alg = pyrads.algms.os_cfar.OSCFAR(
        range_fft_alg.out_data_shape,
        **oscfar_params
    )

    # Create Pipeline instance with the list of defined algorithms
    algorithms = [
        remove_offset_alg,
        window_alg,
        range_fft_alg,
        oscfar_alg]
    pipeline = pyrads.pipeline.Pipeline(algorithms)
    pipe_data = pipeline(reduced_data)
    fft_out = pipe_data[-2]
    out = pipe_data[-1]

    # Plot results
    if multi_ramp==False:
        fft_data = fft_out[frame_n, 0, 0, chirp_n, :]
        out_data = out[frame_n, 0, 0, chirp_n, :]
        pyrads.utils.plotter.plot_single_ramp_pipeline(image, fft_data, out_data)
    else:
        fft_data = fft_out[:, 0, 0, chirp_n, :]
        out_data = out[:, 0, 0, chirp_n, :]
        pyrads.utils.plotter.plot_multi_ramp_pipeline(images, fft_data, out_data)
    return


if __name__ == "__main__":
    main()
