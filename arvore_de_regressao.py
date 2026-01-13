from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegresso
from sklearn.metrics import mean_squared_error, r2_score
# gerando os dados
X = [[0, 0], [1, 1], [2, 2]]
Y = [0, 1, 2]
# criação do modelo com o Arvore de Regressão
clfT = DecisionTreeRegressor(criterion="mse")
clfF = RandomForestRegressor(criterion="mse", n_estimators=10)
# treina o modelo 
clfT.fit (X, Y) 
clfF.fit (X, Y) 
# gera previsões para outros valores
Y_pred_T = clfT.predict(X)
Y_pred_F = clfF.predict(X)
# avalia o erro das previsões
print('MSE Árvore: %.2f', mean_squared_error(Y, Y_pred_T))
print('MSE Floresta: %.2f', mean_squared_error(Y, Y_pred_F))
print('R2: %.2f', r2_score(Y, Y_pred_T))
print('R2: %.2f', r2_score(Y, Y_pred_F))