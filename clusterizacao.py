import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot

datasetO = pd.read_csv('filmes.csv', index_col=0)

# print("{}".format(datasetO))

dataframe = np.array(datasetO[['budget', 'rating']])

# print("{}".format(dataframe))
k = 4

kmeans = KMeans(n_clusters=k)
kmeans.fit(dataframe)

labels = kmeans.labels_
centroides = kmeans.cluster_centers_

# print("Clusters: {}".format(labels))
# print("Centroides: {}".format(centroides))

x_min = dataframe[:, 0].min() - 0.1
y_min = dataframe[:, 1].min() - 0.1

x_max = dataframe[:, 0].max() + 0.1
y_max = dataframe[:, 1].max() + 0.1

x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

pyplot.plot(x_grid, y_grid, marker='.', color='black')
#pyplot.show()

cores_clusters = ['red', 'blue', 'green','gray']
for i in range(k):
    dados_cluster = dataframe[np.where(labels == i)]
    pyplot.plot(dados_cluster[:, 0], dados_cluster[:, 1], 'o', c=cores_clusters[i])
    pyplot.plot(centroides[i, 0], centroides[i, 1], '+', color='purple', markersize=15, markeredgewidth=2)

pyplot.xlabel('Altura Filha')
pyplot.ylabel('Altura Mãe')
pyplot.show()

pred_grid = kmeans.predict(np.c_[x_grid.ravel(), y_grid.ravel()])
print(pred_grid)


pred_grid = pred_grid.reshape(x_grid.shape)
pyplot.imshow(pred_grid, interpolation='nearest', extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()), cmap=pyplot.cm.Paired, aspect='auto', origin='lower')

pyplot.plot(dataframe[:, 0], dataframe[:, 1], 'o', markersize=5, color='white')
pyplot.plot(centroides[:,0], centroides[:,1], '+', color = 'purple', markersize = 15, markeredgewidth = 1)

pyplot.title('Diagrama de Voronoi - Clusterização')
pyplot.xlim(x_min, x_max)
pyplot.ylim(y_min, y_max)
pyplot.show()
