import math

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
            wcss += self.get_sum_squared_difference(cluster_i[self.columns], self.centroids.iloc[i]).sum()

        return wcss

    # def get_cluster_index_of_point(self, i, groups):
    #     pass

    '''
    def __get_silhouette_template(self, data, groups, a, b):
        def s(i):
            cluster_I = self.zget_cluster_of_point(i)
            if len()
        pass




    


    def get_silhouette_score_simplified(self, data, groups):
        def a(i):
            return self.get_euclidean_distance()
        def b(i):
            pass

        return (b_val - a_val) / max(a_val, b_val) if len(cluster_I) > 1 else 0
    '''

    def get_silhouette_score(self, data, groups, simplified=True):
        def s(i):
            nonlocal clusters
            nonlocal I

            # If cluster only contains 1 point, return 0
            # a single point is represented by a Series
            if type(clusters[I]) == pd.Series:
                return 0

            if simplified:
                def a(i):
                    nonlocal I
                    return self.get_euclidean_distance(i, self.centroids.iloc[I])
                def b(i):
                    nonlocal I
                    nearest_neighbor_distance = self.get_euclidean_distance(i, self.centroids.drop(index=I)).min()
                    return nearest_neighbor_distance if not math.isnan(nearest_neighbor_distance) else math.inf
            else:
                def a(i):
                    nonlocal clusters
                    nonlocal I
                    return self.get_euclidean_distance(i, clusters[I]).sum() / (len(clusters[I]) - 1)
                def b(i):
                    nonlocal clusters
                    nonlocal I
                    nearest_neighbor_distance = math.inf
                    for J in range(self.k):
                        if I != J:
                            nearest_neighbor_distance = min(nearest_neighbor_distance, self.get_euclidean_distance(i, clusters[J]).sum() / len(clusters[J]))
                    return nearest_neighbor_distance

            a_val = a(i)
            b_val = b(i)
            return (b_val - a_val) / max(a_val, b_val)

        # #grouped_data = pd.concat([data, groups.rename('group')], axis=1)
        # for I in range(self.k):
        #     cluster_I = grouped_data.loc[grouped_data['group'] == I]

        clusters = [c for _, c in data.groupby(groups)]

        total_score = 0
        for I in range(self.k):
            total_score += clusters[I].apply(s, axis=1).sum()

        average_score = total_score / len(data)
        print(f'Silhouette score with {self.k} clusters: {average_score}')
        return average_score

        #return grouped_data.apply(s).mean()

    def get_silhouette_score_simplified(self, data, groups):
        def s(i):
            I = i['group'].astype(np.int32)

            nonlocal cluster_counts
            # If cluster only contains 1 point, return 0
            if cluster_counts.iloc[I] == 1:
                return 0

            a_val = self.get_euclidean_distance(i, self.centroids.iloc[I])
            b_val = self.get_euclidean_distance(i, self.centroids.drop(index=I)).min()

            return (b_val - a_val) / max(a_val, b_val)

        cluster_counts = groups.value_counts()
        print(cluster_counts)

        grouped_data = pd.concat([data, groups.rename('group')], axis=1)

        #average_score = grouped_data.apply(s, axis=1).mean()
        average_score = 0
        for _, i in grouped_data.iterrows():
            I = i['group'].astype(np.int32)

            # If cluster only contains 1 point, return 0
            if cluster_counts.iloc[I] == 1:
                return 0

            a_val = self.get_euclidean_distance(i, self.centroids.iloc[I])
            b_val = self.get_euclidean_distance(i, self.centroids.drop(index=I)).min()

            average_score += (b_val - a_val) / max(a_val, b_val)
        average_score /= len(data)
        #average_score = np.vectorize(s)(grouped_data, grouped_data['group'].astype(np.int32))
        print(f'Silhouette score with {self.k} clusters: {average_score}')
        return average_score