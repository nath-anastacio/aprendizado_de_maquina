import pandas as pd       
from sklearn.model_selection import train_test_split               

#Arquivo que possui o conjunto de dados a ser treinado                        
original_data = pd.read_csv('mtcars.csv')                    
  

#Método para separar o conjunto de dados em amostra de treino, teste e validação
def data_split(dat,trf = 0.5,vlf=0.25,tsf = 0.25): 
      nrows = dat.shape[0]     
      trnr = int(nrows*trf) 
      vlnr = int(nrows*vlf)    