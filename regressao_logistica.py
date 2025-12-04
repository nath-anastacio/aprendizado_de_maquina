# Importar as bibliotecas necessárias
from sklearn.datasets import load_iris

# Importando o modelo de Regressão Logística (Logistic Regression)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Importando os dados
X, y = load_iris(return_X_y=True)

# Separando base de treino da validação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Criando o classificador
lr = LogisticRegression()

# Treinando o classificador
lr.fit(X_train, y_train)

# Realizando as previsões
y_pred = lr.predict(X_test, y_test)