# Importando as bibliotecas
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split

# Importando o modelo árvore de decisão (Decision Tree Classifier)
from sklearn.tree import DecisionTreeClassifier

# Carregando os dados
X, y = load_iris(return_X_y=True)

# Separando a base de treino da validação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Criando o classificador
dtc = DecisionTreeClassifier()

# Treinando o classificador
dtc.fit(X_train, y_train)

# Realizando as previsões
y_pred = dtc.predict(X_test)