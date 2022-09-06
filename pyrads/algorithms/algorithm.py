#!/usr/bin/env python3
"""
Module containing the base algorithm class
"""
# Standard libraries
from abc import ABC, abstractmethod
# Local libraries

class Algorithm(ABC):
    """
    Parent class for radar algorithms
    """
    def __init__(self):
        self.input_shape = None

    def test_method(self):
        pass
