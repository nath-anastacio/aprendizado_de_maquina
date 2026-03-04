#Etapa 1 - Em todo processamento que envolve aprendizagem de máquina, é necessário selecionar apenas aqueles dados
#que serão utilizados na análise para o treinamento.
#Para isso, serão utilizados os dados: idade, tempo de estudo, faltas e prova 1. 


#Etapa 2 – Treinamento da rede neural 

x = np.rot90(x) 
net = perceptron.Perceptron(max_iter=100,eta0=0.1) 
net.fit(x, y)  

#Etapa 3 – Avaliação do Modelo 

print ('Saída Esperada  ' + str(net.predict(x))) 
print ('Saída Atual     ' + str(y)) 
print ('Precisão        ' + str(net.score(x, y) * 100) + '%') 
print('Pesos: ' + str(net.coef_)) 