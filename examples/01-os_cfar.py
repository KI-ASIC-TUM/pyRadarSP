#!/usr/bin/env python3
"""
Run OS-CFAR over test dataset
"""
# Standard libraries
import numpy as np
# Local libraries
import pyrads.algms.os_cfar
import pyrads.pipeline
import pyrads.pipes.preproc_pipeline
import pyrads.utils.plotter


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
        "ordered_k": 10,
        "alpha": 0.5,
        "n_guard_cells": 8,
    }
    if False:
        data = pyrads.utils.data_loader.load_dataset(**dataset_params)
    data = np.ones((256))
    # Create a pipeline instance and add the data and a signal processing
    # algorithm to it.
    if False:
        preproc_pipe = pyrads.pipes.preprocesing_pipeline.Pipeline(**preproc_params)
    oscfar_alg = pyrads.algms.os_cfar.OSCFAR(**oscfar_params)

    algorithms = [oscfar_alg]
    pipeline = pyrads.pipeline.Pipeline(algorithms)
    pipeline.run(data)

    # Plot data before executing pipeline
    return


if __name__ == "__main__":
    main()
