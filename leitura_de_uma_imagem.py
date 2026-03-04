#Leitura de uma imagem
# Importar bibliotecas
import cv2
import numpy as np
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib


# Leitura da imagem
img = cv2.imread('imagem.png')

 

# Apresentação da imagem na tela
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


##########
#Detecção de faces
#Dimensão das imagens
plt.rcParams['figure.figsize'] = (224, 224)

#Classificador construído para detectar faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Transformando a imagem em escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Processo para detectar as faces na imagem
faces = face_cascade.detectMultiScale(
 gray,
 scaleFactor=1.1,
 minNeighbors=5,
 minSize=(30, 30),
)

cont = 0
for (x,y,w,h) in faces:
 img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
 roi_gray = gray[y:y+h, x:x+w]
 roi_color = img[y:y+h, x:x+w]
 cont=cont + 1

cv2.imwrite('aragorn.png',img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


#########
#Construção de um gráfico de cores
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carrega a imagem com seus dados de altura, largura e canais
# Cria a tupla colors (BGR) e cria a lista features
img = cv2.imread('serie.jpg',-1)
height, width, channels = img.shape
colors = ('b', 'g', 'r')
features=[]

#Cria a máscara
mask = np.zeros(img.shape[:2], np.uint8)
mask[int(height*0.1):int(height*0.9), 0:int(width*0.6)] = 255
masked_img = cv2.bitwise_and(img,img,mask = mask)

#Exemplo Doc OpenCV
#Carrega a imagem com a adição da máscara
for i,col in enumerate(colors):
 hist_mask = cv2.calcHist([img], [i], mask, [256], [0, 256])
 plt.plot(hist_mask, color=col)
 plt.xlim([0, 256])
plt.show()