from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
# gerando os dados
X = [[0, 0], [1, 1], [2, 2]]
Y = [0, 1, 2]
# criação do modelo com a SVR
svr = SVR(C=1.0, epsilon=0.2)
svr.fit(X, Y) 
# gera previsões para outros valores
Y_pred = svr.predict(X)
# avalia o erro das previsões
print('MSE: %.2f', mean_squared_error(Y, Y_pred))
print('R2: %.2f', r2_score(Y, Y_pred))