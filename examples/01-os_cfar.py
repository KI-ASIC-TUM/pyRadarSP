#!/usr/bin/env python3
"""
Run OS-CFAR over test dataset
"""
# Standard libraries
# Local libraries
import radar_processing.utils.plotter

def main():
    params = {
        dataset: "test_dataset_1",
    }
    # Create a pipeline instance and add the data and a signal processing
    # algorithm to it.
    preproc_pipe = radar_processing.pipelines.preprocesing_pipeline.Pipeline()
    pipeline.data = sensor_data
    pipeline.add_processing_algorithm(pyradar.signal_processing.identity)
    pipeline.add_processing_algorithm(pyradar.signal_processing.flatten, 5)

    # Plot data before executing pipeline
    radar_processing.utils.plotter.plot_cfar()
    return


if __name__ == "__main__":
    main()
