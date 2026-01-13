# Importando as bibliotecas
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Carregando os dados
X, y = load_iris(return_X_y=True)

# Separando base de treino da validação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Criando o classificador
clf = SVC(kernel='linear', C=1.0)

# Treinando o classificador
clf.fit(X_train, y_train)

# Realizando as previsões
clf.predict(X_test, y_test)