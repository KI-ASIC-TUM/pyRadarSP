#!/usr/bin/env python3
"""
Module containing the base pipeline class
"""
# Standard libraries
from abc import ABC, abstractmethod
import numpy as np
# Local libraries


class Pipeline():
    """
    Parent class for radar algorithms
    """
    def __init__(self, chain=[], dataset=""):
        self.__name__ == "pyrads.Pipeline"
        self.__algorithms = []
        self.__in_data = np.array([])
        self.output = np.array([])

        if len(dataset)>0:
            self.add_data(dataset)

        if len(chain)>0:
            self.add_chain(chain)


    def add_data(self, name):
        pass


    def __check_shape(self, in_shape):
        """
        Check that input shape matches the last element of the pipeline
        """
        # If there already are algorithms, use the last one as reference
        if len(self.__algorithms) > 0:
            cur_out_data_shape = self.__algorithms[-1].out_data_shape
        # Otherwise, use the input data shape
        elif self.__in_data.size > 0:
            cur_out_data_shape = self.__in_data.shape
        # Otherwise, ignore the check
        else:
            cur_out_data_shape = in_shape
        check = True if in_shape == cur_out_data_shape else False
        return check


    def add_chain(self, chain):
        """
        Add a chain of algorithms or pipelines
        """
        for element in chain:
            if type(element) == "Algorithm":
                self.add_algorithm(element)
            elif type(element) == "Pipeline":
                self.add_chain(element.__algorithms)
            else:
                raise TypeError("input type invalid")


    def add_algorithm(self, algorithm):
        """
        Add an algorithm to the processing chain
        """
        # Test that the input data format is compatible with the previous one
        self.__check_shapes(algorithm.in_data_shape)
        # Add algorithm to the list
        self.__algorithms.append(algorithm)


    def run(self, in_data):
        # Check data shape fits the 1st algorithm in_shape
        pass


    def __call__(self):
        self.run()
