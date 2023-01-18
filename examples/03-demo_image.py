#!/usr/bin/env python3
"""
Generate a sample image for representing the dataset
"""
# Standard libraries
import matplotlib.pyplot as plt
import numpy as np
# Local libraries
import dhandler.h5_handler
import pyrads.algms.fft
import pyrads.algms.os_cfar
import pyrads.algms.remove_offset
import pyrads.algms.window
import pyrads.pipeline


def plotter(raw_out, fft_out, image):
    """
    Plotting function
    """
    fft_no_offset = fft_out[3:150]
    fft_no_offset -= fft_no_offset.min()-0.0001
    fft_no_offset /= 2.1*fft_no_offset.max()
    
    fig, axs = plt.subplots(2, 1, figsize=(12,8))
    plt.subplot(2, 1, 1)
    plt.imshow(image[5:-20,50:])
    plt.xticks([])
    plt.yticks([])
    ax1 = plt.subplot(2,2,3)
    plt.plot(raw_out)
    plt.xticks([])
    ax1.spines[['left','right', 'top']].set_visible(False)
    ax1.set_facecolor('whitesmoke')
    ax1.set_yticklabels([])
    ax1.set_ylim([-0.2, 0.3])
    ax1.yaxis.set_ticks_position('none')
    ax1.grid(axis="y", color="white")
    plt.xlabel("Time (ns)")
    ax2 = plt.subplot(2,2,4)
    plt.plot(fft_no_offset)
    ax2.spines[['left','right', 'top']].set_visible(False)
    plt.grid()
    ax2.set_facecolor('whitesmoke')
    plt.xlabel("Frequency (kHz)")
    plt.xticks([])
    ax2.set_facecolor('whitesmoke')
    ax2.set_ylim([0, 0.5])
    # ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
    ax2.set_yticklabels([])
    ax2.yaxis.set_ticks_position('none')
    ax2.grid(axis="y", color="white")
    fig.tight_layout()
    plt.savefig("scene7_demo.eps", dpi=300)
    plt.savefig("scene7_demo.png", dpi=300)
    plt.show()


def main(frame_n=14, chirp_n=30, multi_ramp=True, scene_n=7):
    """
    Main routine

    Data shape: (n_frames, tx_antennas, rx_antennas, n_ramps, n_samples)

    @multi_ramp: If true, plot a sequence of ramps. If false, only one
        ramp is plotted.
    """
    h5_handler = dhandler.h5_handler.H5Handler("OTH/scene{}_0".format(scene_n))
    data, radar_config, calib_vec = h5_handler.load(
        dataset_dir=None
    )
    images = h5_handler.load_images()
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

    # Create Pipeline instance with the list of defined algorithms
    algorithms = [
        remove_offset_alg,
        window_alg,
        range_fft_alg
        ]
    pipeline = pyrads.pipeline.Pipeline(algorithms)
    pipe_data = pipeline(reduced_data)
    raw_out = pipe_data[2][frame_n, 0, 0, chirp_n, :]
    fft_out = pipe_data[-1][frame_n, 0, 0, chirp_n, :]

    # Plot results
    plotter(raw_out, fft_out, image)
    return


if __name__ == "__main__":
    main()
