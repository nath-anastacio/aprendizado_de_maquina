from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
# gerando os dados
X = [[0, 0], [1, 1], [2, 2]]
Y = [0, 1, 2]
# criação do modelo com o Método MMQ
clf = linear_model.LinearRegression()
# treina o modelo 
clf.fit (X, Y) 
# exibe os coeficientes do modelo
clf.coef_
# gera previsões para outros valores
Y_pred = clf.predict(X)
# avalia o erro das previsões
print('MSE: %.2f', mean_squared_error(Y, Y_pred))
print('R2: %.2f', r2_score(Y, Y_pred))


## Existem outras implementações do OLS disponíveis para a linguagem Python.
#  Uma das mais conhecidas é a StatsModels, que é um pacote Python que permite 
# aos usuários explorar dados, estimar modelos estatísticos e executar testes estatísticos. 

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
# gerando os dados
X = [[0, 0], [1, 1], [2, 2]]
Y = [0, 1, 2]
# criação do modelo com o Método MMQ
clf = linear_model.SGDRegressor(alpha=0.1, n_iter=20)
# treina o modelo 
clf.fit (X, Y) 
# exibe os coeficientes do modelo
clf.coef_
# gera previsões para outros valores
Y_pred = clf.predict(X)
# avalia o erro das previsões
print('MSE: %.2f', mean_squared_error(Y, Y_pred))
print('R2: %.2f', r2_score(Y, Y_pred))
