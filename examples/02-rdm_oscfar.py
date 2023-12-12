#!/usr/bin/env python3
"""
Run RD-map over a dynamic scenario
"""
# Standard libraries
import numpy as np
# Local libraries
import dhandler.oth_handler
import dhandler.h5_handler
import pyrads.algms.fft
import pyrads.algms.os_cfar
import pyrads.algms.remove_offset
import pyrads.algms.window
import pyrads.pipeline
import pyrads.utils.plotter


NEW_SCENES = {
    "new_1": "20230127_114455",
    "new_2": "20230125_131034",
    "new_3": "20230314_132412",
    "new_4": "20230314_133320",
    "new_5": "20230314_132605",
    "new_6": "20230314_132927",
}

def main(frame_n=30, multi_frame=True, overlap=True, scene_n="new_1"):
    """
    Main routine for the 2D OS-CFAR example

    Data shape: (n_frames, tx_antennas, rx_antennas, n_ramps, n_samples)

    @multi_frame: If true, plot a sequence of ramps. If false, only one
        ramp is plotted.
    @overlap: If true, plot the CFAR on top of the FFT plot
    """
    if type(scene_n) is str:
        handler = dhandler.oth_handler.OTHHandler("OTH/"+NEW_SCENES[scene_n])
    else:
        handler = dhandler.h5_handler.H5Handler("OTH/scene{}_0".format(scene_n))

    data, radar_config, calib_vec = handler.load(skip_permission=False)
    objects = handler.load_objects(dataset_dir=None)
    objects = np.array(objects)
    images = handler.load_images(image_types=['image_left_stereo'])
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
        "type": "range-doppler",
        "normalize": True,
        "logarithmic_out": True,
        "out_format": "modulus"
    }
    oscfar_params = {
        "n_dims": 2,
        "window_width": 10,
        "ordered_k": 80,
        "alpha": 0.75,
        "n_guard_cells": 4,
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
        oscfar_alg
        ]
    pipeline = pyrads.pipeline.Pipeline(algorithms)
    pipe_data = pipeline(reduced_data)
    fft_out = pipe_data[-2]
    oscfar_out = pipe_data[-1]

    # Plot results
    if multi_frame==False:
        fft_data = fft_out[frame_n, 0, 0, :, :]
        out_data = oscfar_out[frame_n, 0, 0, :, :]
        pyrads.utils.plotter.plot_rd_map(
                image, fft_data, out_data, scene_n=scene_n)
    else:
        fft_data = fft_out[:, 0, 0, :, :]
        out_data = oscfar_out[:, 0, 0, :, :]
        if overlap:
            n_plots = 2
        else:
            n_plots = 3
        pyrads.utils.plotter.plot_multi_ramp_pipeline(
                images,
                fft_data,
                out_data,
                ndims=2,
                overlap=overlap,
                scene_n=scene_n,
                n_plots = n_plots,
                init_frame=0,
                end_frame=len(images)
        )
    return


if __name__ == "__main__":
    main()
