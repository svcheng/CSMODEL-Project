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

        centroids = grouped_data.groupby('group').median(numeric_only=True).iloc[:, self.start_var:self.end_var]

        return centroids


    def get_wcss(self, data, groups):
        """Returns the within-cluster sum of squares (WCSS) or the inertia
        which is equal to the sum of squared distances from each point to its centroid.
        (equal to the sum OF the sum of squared difference from each point to its centroid)

        Arguments:
            data {DataFrame} -- clustered dataset
            groups {Series} -- represents the cluster of each data point in the
            dataset.
        Returns:
            np.float64 -- contains the value of the inertia of the clustered data.
        """

        wcss = 0
        for i in range(self.k):
            cluster_i = data.loc[groups == i]
            #print(cluster_i)
            #print(cluster_i[self.columns])
            #print(self.centroids.iloc[i])
            val = self.get_sum_squared_difference(cluster_i[self.columns], self.centroids.iloc[i]).sum()
            #print(val)
            wcss += val

        return wcss