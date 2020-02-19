from sklearn.metrics.pairwise import cosine_distances
class custom_KMeans:
    """The k-means algorithm"""

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def train(self, df):
        df = pd.DataFrame(df)
        self.data = df.copy()
        self.clusters = np.zeros(self.data.shape[0])

        ### initialize centroids
        rows = self.data
        rows.reset_index(drop=True, inplace=True)
        self.centroids = rows.sample(n=self.n_clusters)
        self.centroids.reset_index(drop=True, inplace=True)

        # Initialize old centroids as all zeros
        self.old_centroids = pd.DataFrame(np.zeros(shape=(self.n_clusters, self.data.shape[1])),
                                          columns=self.data.columns)

        # check the distance of each data point to the centroid and assigning each point to the closest cluster.
        while not self.old_centroids.equals(self.centroids):
            # Stash old centroids
            self.old_centroids = self.centroids.copy(deep=True)

            # Iterate through each data point/set
            for row_i in range(0, len(self.data)):
                distances = list()
                point = self.data.iloc[row_i]

                # Calculate the distance between the point and centroid
                # I'll be using cosine similarity here
                for row_c in range(0, len(self.centroids)):
                    centroid = self.centroids.iloc[row_c]
                    point_array = np.array(point).reshape(1,-1)
                    centroid_array = np.array(centroid).reshape(1,-1)
                    distances.append(cosine_distances(point_array, centroid_array)[0][0])

                # Assign this data point to a cluster
                self.clusters[row_i] = int(np.argmin(distances))

            # For each cluster extract the values which now belong to each cluster and calculate new k-means
            for label in range(0, self.n_clusters):

                label_idx = np.where(self.clusters == label)[0]

                if len(label_idx) == 0:
                    self.centroids.loc[label] = self.old_centroids.loc[label]
                else:
                    # Set the new centroid to the mean value of the data points within this cluster
                    self.centroids.loc[label] = self.data.iloc[label_idx].mean()
