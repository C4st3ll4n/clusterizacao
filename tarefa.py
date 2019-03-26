import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot


class Clusterizador:

    def __init__(self):

        self.dataset = None
        self.data = None
        self.k = 3
        self.kmeans = KMeans(self.k)
        self.labels = None  # self.kmeans.labels_
        self.centroides = None  # self.kmeans.cluster_centers_

    def processar(self, x, y):
        x_min = self.data[:, 0].min() - 0.1
        y_min = self.data[:, 1].min() - 0.1

        x_max = self.data[:, 0].max() + 0.1
        y_max = self.data[:, 1].max() + 0.1

        x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

        # Executa a predição com o modelo kmeans
        pred_grid = self.kmeans.predict(np.c_[x_grid.ravel(), y_grid.ravel()])

        # Reshape do resultado da predição para as dimensões do espaço definido
        pred_grid = pred_grid.reshape(x_grid.shape)
        # Plot dos polígonos de Voronoi
        pyplot.imshow(pred_grid, interpolation='nearest',
                      extent=(x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()), cmap=pyplot.cm.Paired,
                      aspect='auto', origin='lower')

        cores_clusters = ['red', 'blue', 'green']
        for i in range(self.k):
            dados_cluster = self.data[np.where(self.labels == i)]
            pyplot.plot(dados_cluster[:, 0], dados_cluster[:, 1], 'o', c=cores_clusters[i])
            pyplot.plot(self.centroides[i, 0], self.centroides[i, 1], '*', color=cores_clusters[i], markersize=15,
                        markeredgewidth=2)

        pyplot.title('Diagrama de Voronoi - Clusterização')
        pyplot.xlabel(x)
        pyplot.ylabel(y)
        pyplot.xlim(x_min, x_max)
        pyplot.ylim(y_min, y_max)
        pyplot.show()

    def input_k(self):
        con = False
        while not con:

            k = int(input("Insira a quantidade de clusters: "))
            if k <= 0:
                print("\nOpção inválida, tente novamente !\n")
            else:
                con = True
                return k

    def dataset1(self):

        self.dataset = pd.read_csv('filmes.csv', index_col=0)
        self.data = np.array(self.dataset[['budget', 'rating']])

        self.kmeans.fit(self.data)

        self.labels = self.kmeans.labels_
        self.centroides = self.kmeans.cluster_centers_
        print("Labels: {}\nCentroide: {}\n".format(self.labels, self.centroides))

        self.processar("Orçamento", "Avaliação")

    def dataset2(self):

        self.dataset = pd.read_csv('seed.csv', index_col=0)
        self.data = np.array(self.dataset[["asymmetry_coefficient", "classification"]])
        self.kmeans.fit(self.data)

        self.labels = self.kmeans.labels_
        self.centroides = self.kmeans.cluster_centers_

        self.processar("Constante de Asímetria", "Classificação")

    def dataset3(self):

        self.dataset = pd.read_csv('eruptions.csv', index_col=0)
        self.data = np.array(self.dataset[["eruptions", "waiting"]])

        self.kmeans.fit(self.data)

        self.labels = self.kmeans.labels_
        self.centroides = self.kmeans.cluster_centers_

        self.processar("Erupções", "Waiting")

    def clusterizar(self, opcao):
        print("Você escolheu a opção:{}".format(opcao))
        print("\nAguarde, enquanto processamos os dados .....\n")
        if opcao == 1:
            self.dataset1()
        if opcao == 2:
            self.dataset2()
        if opcao == 3:
            self.dataset3()

        # print(self.dataset)
        # print(self.data)
        # print("Centroides: {}\n".format(self.centroides))


def tratar_input():
    condition = False
    while not condition:
        op = int(input("Qual dataset você deseja clusterizar ?"
                       "\n1 - Filmes\n2 - Sementes ou\n3 - Erupções vulcanicas\nOpção escolhida: "))

        if op != 1:
            if op != 2:
                if op != 3:
                    print("\nOpção inválida, tente novamente !\n")
                else:
                    condition = True
                    return op
            else:
                condition = True
                return op
        else:
            condition = True
            return op


c = Clusterizador()
c.clusterizar(tratar_input())
