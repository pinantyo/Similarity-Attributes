import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random
import pickle


def custom_euclidean(point, data, penalty=False):
    results = point - data

    if penalty:
      return np.sqrt(np.sum(((results)**2)*penalty, axis=1))

    return np.sqrt(np.sum((results)**2, axis=1))


class KMEANS:
  def __init__(self, cluster = 2, max_iter=300, penalty=[]):
    self.n_clusters = cluster
    self.max_iter = max_iter
    self.penalty = penalty

  def fit(self, data_train):      
    min_, max_ = np.min(data_train, axis=0), np.max(data_train, axis=0)
    self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]       
    iteration = 0
    prev_centroids = None
    while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:

      sorted_points = [[] for _ in range(self.n_clusters)]
      for x in data_train:
        dists = custom_euclidean(x, self.centroids, self.penalty)
        centroid_idx = np.argmin(dists)
        sorted_points[centroid_idx].append(x)           
      prev_centroids = self.centroids
      self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
      for i, centroid in enumerate(self.centroids):
        if np.isnan(centroid).any():  
          self.centroids[i] = prev_centroids[i]
      iteration += 1
  
  def evaluate(self, X):
    centroids = []
    centroid_idxs = []
    for x in X:
      dists = custom_euclidean(x, self.centroids, self.penalty)
      centroid_idx = np.argmin(dists)
      centroids.append(self.centroids[centroid_idx])
      centroid_idxs.append(centroid_idx)        
    
    return centroids, centroid_idxs, dists


if __name__ == '__main__':

  # N Cluster
  centers = 4

  # Get Data
  df = pd.read_excel('/content/dummy.xlsx')

  # Normalization
  df[['NPWP','Akte Pendirian','Tanggal']] = StandardScaler().fit_transform(df[['NPWP','Akte Pendirian','Tanggal']])


  # Feature Extraction
  tf_idf = TfidfVectorizer(
      lowercase=True, 
      ngram_range=(1, 1)
  )

  # Convert Probs to Dataframe
  nama = pd.DataFrame(
      tf_idf.fit_transform(df['Nama']).toarray(),
      columns=list(tf_idf.vocabulary_.keys())
  )


  # Merging

  df = pd.merge(df, nama, left_index=True, right_index=True)

  # Fitting
  kmeans = KMEANS(cluster=centers, max_iter=5000, penalty=[0.3, 0.3, 0.1] + [0.3]*101)
  kmeans.fit(df.values)

  pickle.dump(kmeans, open('custom_kmean.pkl','wb'))


  kmeans = pickle.load(open('custom_kmean.pkl','rb'))

  # Clustering
  class_centers, classification, distance = kmeans.evaluate(df.values)

  df['Cluster'] = classification

  df.tocsv("clustered.csv")