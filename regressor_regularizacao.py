from sklearn.linear_model import Lasso 
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet 
from sklearn.metrics import mean_squared_error, r2_score
# gerando os dados
X = [[0, 0], [1, 1], [2, 2]]
Y = [0, 1, 2]
# Lasso - criação da Regressão Lasso com parâmetro lambda
lasso = Lasso(alpha=0.1)
lasso.fit(X, Y)
y_pred_lasso = lasso.predict(X_test)
r2_score_lasso = r2_score(Y, y_pred_lasso)

# Ridge
ridge = Ridge(alpha=0.1) 
y_pred_ridge = ridge.fit(X, Y)
y_pred_ridge = ridge.predict(X) 
r2_score_ridge = r2_score(Y, y_pred_ridge) 

# ElasticNet
enet = ElasticNet(alpha=0.1, l1_ration=0.7)
y_pred_enet = enet.fit(X, Y)
y_pred_enet = enet.predict(X)
r2_score_enet = r2_score(Y, y_pred_enet)

print "r^2 enet : %f" % r2_score_enet
print "r^2 lasso : %f" % r2_score_lasso 
print "r^2 ridge : %f" % r2_score_ridge