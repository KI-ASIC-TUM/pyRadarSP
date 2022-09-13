#!/usr/bin/env python3
"""
Run OS-CFAR over test dataset
"""
# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
# Local libraries
import dhandler.h5_handler
import dhandler.npy_handler
import pyrads.algms.fft
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
    data, radar_config, calib_vec = h5_handler.load(dataset_params["dataset"], dataset_dir=None)
    data = data[0]

    # User-defined parameters for os-cfar algorithm
    einsum_window_params = {
        "axis": -1,
        "window_type": "hann"
    }
    fft_params = {
        "type": "range",
    }
    # Create a pipeline instance and add the data and a signal processing
    # algorithm to it.

    remove_offset_alg = pyrads.algms.remove_offset.RemoveOffset(in_data_shape=data.shape)
    window_alg = pyrads.algms.window.Window(in_data_shape=remove_offset_alg.out_data_shape, **einsum_window_params)
    range_fft_alg = pyrads.algms.fft.FFT(in_data_shape=window_alg.out_data_shape, **fft_params)

    algorithms = [remove_offset_alg, window_alg, range_fft_alg]
    pipeline = pyrads.pipeline.Pipeline(algorithms)
    out = pipeline(data)

    fig, ax = plt.subplots(nrows=4)
    ax[0].plot(out[0][0,0,0,:])
    ax[1].plot(out[1][0,0,0,:])
    ax[2].plot(out[2][0,0,0,:])
    ax[3].plot(out[3][0,0,0,:])
    plt.show()
    
    dh_processed = dhandler.npy_handler.NPyHandler()
    directory="Processed/HiRes/range_fft_data"
    hires_dataset_config = h5_handler.load_config()
    config = dh_processed.save_dataset(dataset_name="range_fft/scenario1_2",
                              data_dict={"data-files": [{"data": out[3], "filename-suffix": "_test_range_fft", "directory":directory}],
                                         "radar-config": [{"data": radar_config, "filename-suffix": "_test_range_fft", "directory":directory}]},
                              scenario_name=None,
                              scene_description=None,
                              data_description=None,
                              source_database_entry=hires_dataset_config[dataset_params["dataset"]],
                              save_config=True)




if __name__ == "__main__":
    main()
