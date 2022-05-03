import numpy as np
from matplotlib import pyplot as plt
import cv2 #OpenCV
import imutils

#Se obtienen las imagenes originales.
img1 = cv2.imread('Izquierda.PNG',1)  #Primera Imagen
img2 = cv2.imread('Derecha.PNG',1)  #Segunda Imagen

# Escalando una imagenes usando imutils.resize
rimg1 = imutils.resize(img1,height=250)
rimg2 = imutils.resize(img2,height=250)

# De BGR a RGB
RGBimg1 = cv2.cvtColor(rimg1, cv2.COLOR_BGR2RGB)
RGBimg2 = cv2.cvtColor(rimg2, cv2.COLOR_BGR2RGB)

#Valores para Matrices
fil = RGBimg1.shape[0] #Filas
col = RGBimg1.shape[1] #Columnas
dim = RGBimg1.shape[2] #Dimencion

#####################  Funcion  ##########################################################################################################################################################

def Impresion(operacion, namme):
    figura, ax = plt.subplots(4, 3)#(Filas, Columnas)
    figura.set_size_inches(12, 8) #Tamaño del recuadro
    colors = ('b', 'g', 'r') #Definimos Colores

    #Ecualizar imagenes
    ECimg1 = cv2.cvtColor(RGBimg1, cv2.COLOR_BGR2YUV)
    ECimg1[:, :, 0] = cv2.equalizeHist(ECimg1[:, :, 0])
    ECimg1 = cv2.cvtColor(ECimg1, cv2.COLOR_YUV2BGR)

    ECop = cv2.cvtColor(operacion, cv2.COLOR_BGR2YUV)
    ECop[:, :, 0] = cv2.equalizeHist(ECop[:, :, 0])
    ECop = cv2.cvtColor(ECop, cv2.COLOR_YUV2BGR)

    ECimg2 = cv2.cvtColor(RGBimg2, cv2.COLOR_BGR2YUV)
    ECimg2[:, :, 0] = cv2.equalizeHist(ECimg2[:, :, 0])
    ECimg2 = cv2.cvtColor(ECimg2, cv2.COLOR_YUV2BGR)

    #Impresion
    ax[0, 0].imshow(RGBimg1)
    ax[0, 0].set_title('Izquierda')
    ax[0, 0].axis('off')
    ax[3, 0].imshow(ECimg1)
    ax[3, 0].axis('off')

    ax[0, 1].imshow(operacion)
    ax[0, 1].set_title(namme)
    ax[0, 1].axis('off')
    ax[3, 1].imshow(ECop)
    ax[3, 1].axis('off')

    ax[0, 2].imshow(RGBimg2)
    ax[0, 2].set_title('Derecha')
    ax[0, 2].axis('off')
    ax[3, 2].imshow(ECimg2)
    ax[3, 2].axis('off')

    #Histogramas
    for i, color in enumerate(colors):
        hist1 = cv2.calcHist([RGBimg1], [i], None, [256], [0, 256])
        ax[1, 0].plot(hist1, color=color)

    for i, color in enumerate(colors):
        hist2 = cv2.calcHist([operacion], [i], None, [256], [0, 256])
        ax[1, 1].plot(hist2, color=color)

    for i, color in enumerate(colors):
        hist3 = cv2.calcHist([RGBimg2], [i], None, [256], [0, 256])
        ax[1, 2].plot(hist3, color=color)

    #Histogramas Ecualizados
    for i, color in enumerate(colors):
        ehist1 = cv2.calcHist([ECimg1], [i], None, [256], [0, 256])
        ax[2, 0].plot(ehist1, color=color)

    for i, color in enumerate(colors):
        ehist2 = cv2.calcHist([ECop], [i], None, [256], [0, 256])
        ax[2, 1].plot(ehist2, color=color)

    for i, color in enumerate(colors):
        ehist3 = cv2.calcHist([ECimg2], [i], None, [256], [0, 256])
        ax[2, 2].plot(ehist3, color=color)
    plt.show()

#####################  Operaciones  ##########################################################################################################################################################
#--------------------------Suma---------------------------------------------------------#
suma = RGBimg1+RGBimg2
namme = 'Suma'
Impresion(suma,namme)
#--------------------------Resta--------------------------------------------------------#
resta = RGBimg1-RGBimg2
namme = 'Resta'
Impresion(resta,namme)
#--------------------------Division-----------------------------------------------------#
with np.errstate(divide='ignore'):
    division = RGBimg1/RGBimg2
    division = division.astype(np.uint8)
    namme = 'Division'
    Impresion(division,namme)
#--------------------------Multiplicacion-----------------------------------------------#
multiplicacion = RGBimg1*RGBimg2
namme = 'Multiplicacion'
Impresion(multiplicacion,namme)
#--------------------------Logaritmo Natural--------------------------------------------#
with np.errstate(divide='ignore'):
     logaritmo_ni = np.log(RGBimg1)
     logaritmo_ni = logaritmo_ni.astype(np.uint8)
     namme = 'Logaritma Izquierda'
     Impresion(logaritmo_ni,namme)
     #------------------------#
     logaritmo_nd = np.log(RGBimg2)
     logaritmo_nd = logaritmo_nd.astype(np.uint8)
     namme = 'Logaritma Derecha'
     Impresion(logaritmo_nd,namme)

#--------------------------Raiz---------------------------------------------------------#
raizi = RGBimg1**(1/2)
raizi = raizi.astype(np.uint8)
namme = 'Raiz Izquierda'
Impresion(raizi,namme)
#------------------------#
raizd = RGBimg2**(1/2)
raizd = raizd.astype(np.uint8)
namme = 'Raiz Derecha'
Impresion(raizd,namme)
#--------------------------Derivada-----------------------------------------------------#
mr = np.empty([fil, col])
mg = np.empty([fil, col])
mb = np.empty([fil, col])
for i in range(fil):
    for j in range(col):
              mr[i][j] = RGBimg1[i][j][0]
              mg[i][j] = RGBimg1[i][j][1]
              mb[i][j] = RGBimg1[i][j][2]
mr = np.diff(mr)
mg = np.diff(mg)
mb = np.diff(mb)
f = mr.shape[0] #Filas
c = mr.shape[1] #Columnas
derivadai = np.zeros((f, c, dim))
for i in range(f):
    for j in range(c):
        derivadai[i][j][0] = mr[i][j]
        derivadai[i][j][1] = mg[i][j]
        derivadai[i][j][2] = mb[i][j]
derivadai = derivadai.astype(np.uint8)
namme = 'Derivada Izquierda'
Impresion(derivadai,namme)
#------------------------#
mr = np.empty([fil, col])
mg = np.empty([fil, col])
mb = np.empty([fil, col])
for i in range(fil):
    for j in range(col):
              mr[i][j] = RGBimg2[i][j][0]
              mg[i][j] = RGBimg2[i][j][1]
              mb[i][j] = RGBimg2[i][j][2]
mr = np.diff(mr)
mg = np.diff(mg)
mb = np.diff(mb)
f = mr.shape[0] #Filas
c = mr.shape[1] #Columnas
derivadad = np.zeros((f, c, dim))
for i in range(f):
    for j in range(c):
        derivadad[i][j][0] = mr[i][j]
        derivadad[i][j][1] = mg[i][j]
        derivadad[i][j][2] = mb[i][j]
derivadad = derivadad.astype(np.uint8)
namme = 'Derivada Derecha'
Impresion(derivadad,namme)
#--------------------------Potencia-----------------------------------------------------#
potenciai = RGBimg1**3
namme = 'Potencia Izquierda'
Impresion(potenciai,namme)
potenciad = RGBimg2**3
#------------------------#
namme = 'Potencia Derecha'
Impresion(potenciad,namme)
#--------------------------Conjunción---------------------------------------------------#
conjuncion = RGBimg1 & RGBimg2
namme = 'Conjunción'
Impresion(conjuncion,namme)
#--------------------------Disyunción---------------------------------------------------#
disyuncion = RGBimg1 | RGBimg2
namme = 'Disyunción'
Impresion(disyuncion,namme)
#--------------------------Negación-----------------------------------------------------#
negacioni = ~RGBimg1
namme = 'Negación Izquierda'
Impresion(negacioni,namme)
negaciond = ~RGBimg2
#------------------------#
namme = 'Negación Derecha'
Impresion(negaciond,namme)
#--------------------------Traslación---------------------------------------------------#
M = np.float32([[1,0,100],[0,1,150]])
traslacioni = cv2.warpAffine(RGBimg1,M,(col,fil))
namme = 'Traslación Izquierda'
Impresion(traslacioni,namme)
#------------------------#
M = np.float32([[1,0,50],[0,1,50]])
traslaciond = cv2.warpAffine(RGBimg2,M,(col,fil))
namme = 'Traslación Derecha'
Impresion(traslaciond,namme)
#--------------------------Escalado-----------------------------------------------------#
escaladoi = cv2.resize(RGBimg1, (0,0), fx=0.4, fy=0.4)
namme = 'Escalado Izquierda'
Impresion(escaladoi,namme)
escaladod = cv2.resize(RGBimg2, (0,0), fx=0.1, fy=0.1)
namme = 'Escalado Derecha'
Impresion(escaladod,namme)
#--------------------------Rotación-----------------------------------------------------#
M = cv2.getRotationMatrix2D((col//2,fil//2),15,1)
rotacioni = cv2.warpAffine(RGBimg1,M,(col,fil))
namme = 'Rotación Izquierda'
Impresion(rotacioni,namme)
#------------------------#
M = cv2.getRotationMatrix2D((col//2,fil//2),40,1)
rotaciond = cv2.warpAffine(RGBimg2,M,(col,fil))
namme = 'Rotación Derecha'
Impresion(rotaciond,namme)
#--------------------------Traslación A fin---------------------------------------------#
pts1 = np.float32([[100,400],[400,100],[100,100]])
pts2 = np.float32([[50,300],[400,200],[80,150]])
M = cv2.getAffineTransform(pts1,pts2)
trasai = cv2.warpAffine(RGBimg1,M,(col,fil))
namme = 'Traslación A fin Izquierda'
Impresion(trasai,namme)
#------------------------#
pts1 = np.float32([[100,400],[400,100],[100,100]])
pts2 = np.float32([[50,300],[400,200],[180,150]])
M = cv2.getAffineTransform(pts1,pts2)
trasad = cv2.warpAffine(RGBimg2,M,(col,fil))
namme = 'Traslación A fin Derecha'
Impresion(trasad,namme)
#--------------------------Transpuesta--------------------------------------------------#
transi = np.zeros((col, fil, dim))
for i in range(fil):
    for j in range(col):
        for k in range (dim):
            transi[j][i][k] = RGBimg1[i][j][k]
transi = transi.astype(np.uint8)
namme = 'Transpuesta Izquierda'
Impresion(transi,namme)
#------------------------#
transd = np.zeros((col, fil, dim))
for i in range(fil):
    for j in range(col):
        for k in range (dim):
            transd[j][i][k] = RGBimg2[i][j][k]
transd = transd.astype(np.uint8)
namme = 'Transpuesta Derecha'
Impresion(transd,namme)
