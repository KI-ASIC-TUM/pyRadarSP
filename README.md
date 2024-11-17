# Python radar signal processing

The *pyRadarSP* library provides an interface for creating and connecting signal
processing algorithms for automotive radar applications in a pipeline manner.

The goal of *pyRadarSP* is to connect data and algorithms from different sources
in a flexible manner, and to provide a basic structure for comparing results.


## Installation

The provided examples use as input data different versions of the automotive
dataset generated within the KI-ASIC project. For easily accessing this data,
we recommend to install the python library that handles and organizes the dataset:

```bash
pip install git@gitlab.lrz.de:ki-asic/pyradarsp.git
```

DISCLAIMER: At the moment of publishing this, the *kiasic-datasets* library and the dataset associated to it was not made public, so any functionality that depends on this library will not be available.

In case that you plan to contribute to *pyRadarSP* or to add your custom algorithms
to the library, we recommend to install it in developer mode:

```bash
git clone git@gitlab.lrz.de:ki-asic/pyradarsp.git
pip install -e ./pyradarsp
```




## Getting started

We recommend you to go to the *examples* folder for understanding the basic
functionality of the library.

The main idea is to keep the library simple. Therefore, we plan to only include
algorithms that are considered fundamental for radar signal processing.

If you plan to develop and test novel algorithms, we recommend to use *pyRadarSP*
for creating the interface and general structure of the signal processing chain,
and to implement in external repositories your own algorithms and pipelines as
children of the provided `Pipeline()` and `Algorithm()` classes.

For example, imagine that you want to develop a new algorithm called
`new_algorithm` that takes as input the data provided by the FFT.
You should first implement `my_algorithm` as a child of the `Algorithm()` class:

```python
#!/usr/bin/env python3
"""
my_algorithm implementation. Its purpose is to bla bla
"""
# Local libraries
import pyrads.algorithm


class MyAlgorithm(pyrads.algorithm.Algorithm):
    """
    Class for implementing my_algorithm
    """
    NAME = "my_algorithm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load os-cfar parameters


    def calculate_out_shape(self):
        """
        The shape of the output data as a function of the input data
        """
        self.out_data_shape = do_something()


    def my_algorithm(self, in_data):
        """
        Main routine of my_algorithm
        """
        # Do something here
        variable_A = do(in_data)
        result = foo(variable_A)
        return result


    def _run(self, in_data):
        """
        Calculate and return result data
        """
        result = my_algorithm(in_data)
        return result
```

Afterwards, you should create a script that integrates `my_algorithm` with the
data processed by the FFT.
There are several ways of doing this. If `my_algorithm` has to be integrated on
a larger pipeline, we recommend combining all preprocessing algorithms in a
secondary pipeline and then combina its output with `my_algorithm` on a higher
level pipeline. Alternatively, you can integrate all algorithms together in a
single pipeline. The structure of this latter approach is simpler, as we show
in the following example:


```python
#!/usr/bin/env python3
"""
Run my_algorithm over test dataset
"""
# Standard libraries
import numpy as np
# Local libraries
import my_algorithm
import dhandler.h5_handler
import pyrads.algms.fft
import pyrads.pipeline


def main():
    # User-defined parameters for the pipeline
    dataset_params = {
        "dataset": "raw_data/scenario1_0",
    }
    h5_handler = dhandler.h5_handler.H5Handler()
    data, _, _ = h5_handler.load(
        dataset_params["dataset"],
        dataset_dir=None
    )
    # Specify the parameters for the employed algorithms
    fft_params = {
        "type": "range",
        "normalize": True,
        "out_format": "modulus"
    }
    my_algorithm_params = {
        "param_A": 1,
        "param_B": True,
    }
    # Create an instance of each of the algorithms
    range_fft_alg = pyrads.algms.fft.FFT(
        data.shape,
        **fft_params
    )
    my_algorithm_alg = my_algorithm.MyAlgorithm(
        range_fft_alg.out_data_shape,
        **my_algorithm_params
    )
    # Create Pipeline instance with the list of defined algorithms
    algorithms = [
        range_fft_alg,
        my_algorithm_alg
    ]
    # Create a Pipeline instance with the algorithms and run it with the data
    pipeline = pyrads.pipeline.Pipeline(algorithms)
    pipe_data = pipeline(data)
    my_alg_out = pipe_data[-1]
    return my_alg_out


if __name__ == "__main__":
    main()
```



## Troubleshooting

Please be aware that the library is currently under development. It is likely
that you will encounter bugs or functionalities not implemented yet.
If you have any feedback on the library or have a specific feature in mind
that you think it should be included, do not hesitate to contact us:

* lopez.randulfe@tum.de
* nico.reeb@tum.de
