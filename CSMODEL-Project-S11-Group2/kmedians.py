import numpy as np
import pandas as pd
from kmeans import KMeans

class KMedians(KMeans):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def adjust_centroids(self, data, groups):
        """Returns the new values for each centroid. This function adjusts
        the location of centroids based on the median of the values of the
        data points in their corresponding clusters.

        Arguments:
            data {DataFrame} -- dataset to cluster
            groups {Series} -- represents the cluster of each data point in the
            dataset.
        Returns:
            DataFrame -- contains the values of the adjusted location of the
            centroids.
        """

        grouped_data = pd.concat([data, groups.rename('group')], axis=1)

        centroids = grouped_data.groupby('group').median(numeric_only=False).iloc[:, self.start_var:self.end_var]

        return centroids