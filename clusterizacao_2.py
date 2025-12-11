from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot

# criando o dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2,
n_redundant=0, n_clusters_per_class=1, random_state=4)

# definindo o modelo
model = AgglomerativeClustering(affinity='euclidean', n_clusters=3, linkage='single')

# treinando o modelo
model.fit(X)

# avaliando uma nova amostra (predição de cluster)
yhat = model.predict(X)

# recuperando os clusters
clusters = unique(yhat)

# Criando um gráfico para exibir os resultados
for cluster in clusters:
    
    # filtrando as amostras de um mesmo cluster
    row_ix = where(yhat == cluster)

    # plotando no gráfico
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])

# exibindo o resultado
pyplot.show()