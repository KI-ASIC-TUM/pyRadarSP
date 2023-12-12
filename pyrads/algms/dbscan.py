#!/usr/bin/env python3
"""
Identity algorithm. Used mostly for debugging
"""
# Standard libraries
import numpy as np
# Local libraries
import pyrads.algorithm


class DBSCAN(pyrads.algorithm.Algorithm):
    """
    Parent class for radar algorithms
    """
    NAME = "DBSCAN"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_dims = kwargs.get("n_dims")
        # Load dbscan parameters
        self.min_pts = kwargs.get("min_pts")
        self.epsilon = kwargs.get("epsilon")
        # dbscan variables
        self.n_clusters = 0


    def calculate_out_shape(self):
        """
        Do not alter the data dimensionality
        """
        self.out_data_shape = self.in_data_shape

    def _run_1d(self, in_data):
        detections = np.where(in_data)[0]
        neighbours = np.zeros_like(detections)
        core_points = np.zeros_like(detections)
        labelled_data = np.zeros_like(in_data) - 1
        # Iterate over all points to count how many neighbours they have
        for i, ego_point in enumerate(detections):
            neighbour_coords = []
            # Iterate over all points right of the ego_point.
            # Neighbours to its left were already counted on previous iterations
            for j, point in enumerate(detections[i+1:]):
                if np.abs(point - ego_point) < self.epsilon:
                    neighbours[i] += 1
                    neighbours[i+j+1] += 1
                    neighbour_coords.append(point)
                # If point is further than epsilon, no more points can be
                # found closer than epsilon
                else:
                    break
            # Determine if ego point is a core points,
            # based on the number of neighbours it has
            core_points[i] = neighbours[i] >= self.min_pts
            # create labelled clusters
            if core_points[i]:
                # Check if point was labelled before
                if labelled_data[ego_point] == -1:
                    self.n_clusters += 1
                    labelled_data[ego_point] = self.n_clusters
                for point in neighbour_coords:
                    labelled_data[point] = labelled_data[ego_point]
            else:
                # Noise is labelled as zero
                labelled_data[ego_point] = 0
        return labelled_data

    def _run(self, in_data):
        """
        Return input data
        """
        if self.n_dims == 1:
            labels = self._run_1d(in_data)
        result = labels
        return result
