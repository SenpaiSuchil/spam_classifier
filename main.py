import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from MLP_batch import *
import random
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from keras.preprocessing.sequence import pad_sequences
#----------extraccion de datos----------#
# Directorio que contiene los archivos de correo electrónico
spam_dir = "spam\\"
easy_ham_dir="easy_ham\\"

#directorio de los archivos que serviran de testeo porque por alguna razon sklear no me quiere dividir en conjuntos
spam_test = "spam_test\\"
not_spam_test = "not_spam_test\\"

# Listas para almacenar el contenido de los correos electrónicos


def mail_extractor(dir,limit):
    files= os.listdir(dir)
    max_files=files[:limit]
    mails=[]
    for archivo in max_files:
        if archivo.endswith(".txt"):
            with open(os.path.join(dir, archivo), "r", encoding="utf-8",errors="ignore") as file:
                contenido = file.read()
                mails.append(contenido)
    return mails

def mail_mixer(array1, array2):
    emails = [(correo, 1) for correo in array1] + [(correo, 0) for correo in array2]
    #mezclamos spam con no spam con esta función para que queden revueltos para entrenar a la red
    random.shuffle(emails)
    

    #este ciclo extrae los correos de la posicion 0 del arreglo emails para almacenarlo en la variable correos
    correos = [correo[0] for correo in emails]
    #lo mismo pero en la pos 1 del arreglo para extraer su etiqueta
    etiquetas = np.array([[correo[1] for correo in emails]])

    return correos,etiquetas

#arreglos corresponidientes de cada conjunto, para entrenamiento y pruebas
spam =mail_extractor(spam_dir,550)#400
easy_ham=mail_extractor(easy_ham_dir,550)#400



spam_mails_test=mail_extractor(spam_test,400)
not_spam_mails_test=mail_extractor(not_spam_test,400)


correos,etiquetas=mail_mixer(spam,easy_ham)
mail_test,test_labels=mail_mixer(spam_mails_test,not_spam_mails_test)



#----------CREACION DE LA BOLSA DE PALABRAS Y ETIQUETADO----------#
# Crear un vectorizador BoW
vectorizer = CountVectorizer()
# Aplicar BoW a los correos electrónicos
X_bow =vectorizer.fit_transform(correos)
test_bow=vectorizer.fit_transform(mail_test)
#transformamos a arreglo
#(no lo mas optimo pero la matriz disperza por alguna extraña razon no me dejaba operar)
X=X_bow.toarray()
X_test=test_bow.toarray()
#transponemos para que las dimensiones cuadren, oremos por que esto funcione, al parecer si entrena
# X=X.T
# X_test=X_test.T
print(X.shape)
print(X_test.shape)

#input()

#maxlen_promedio = (X.shape[0] + X_test.shape[0]) // 2

X_padded = pad_sequences(X, maxlen=30000, padding='pre', truncating='pre')
X_test_padded = pad_sequences(X_test, maxlen=30000, padding='post', truncating='post')

print(X_padded.shape)
print(X_test_padded.shape)


del X,X_test,X_bow,test_bow

# Calcular el mínimo y el rango
minimo = np.min(X_padded)
rango = np.max(X_padded) - minimo

minimo_test = np.min(X_test_padded)
rango_test = np.max(X_test_padded) - minimo

# Aplicar la normalización Min-Max
X_padded = (X_padded - minimo) / rango
X_test_padded = (X_test_padded - minimo_test) / rango_test



X_padded=X_padded.T
X_test_padded=X_test_padded.T




#----------CREACION DE LA RED----------#
#creamos la red: entradas de la cantidad de filas de X, dos capas de 32 neuronas y una salida
net=MLP((X_padded.shape[0],8,4,2,1))

begin=time.time()#tiempo inicial
hora_inicio = datetime.now().time()
print(f"Hora de inicio: {hora_inicio.strftime('%H:%M:%S')}")

net.fit(X_padded,etiquetas)

hora_fin = datetime.now().time()
print(f"Hora de finalización: {hora_fin.strftime('%H:%M:%S')}")

end=time.time()#tiempo cuando sale de fit

print(f"tiempo total de entrenamiento: {end-begin:.2f}")#check para ver cuanto tardó

#inicio del testeo:

y_pred_log=net.predict(X_test_padded)
# Establecer el umbral
print(y_pred_log)

umbral = 0.5

# Etiquetado binario utilizando el umbral
y_pred = (y_pred_log > umbral).astype(int)

# Calcular precisión (accuracy)
accuracy = accuracy_score(test_labels[0], y_pred[0])
print(f'Accuracy: {accuracy:.4f}')

# Calcular puntaje F1
f1 = f1_score(test_labels[0], y_pred[0])
print(f'F1 Score: {f1:.4f}')

# Calcular matriz de confusión
conf_matrix = confusion_matrix(test_labels[0], y_pred[0])
print('Matriz de Confusión:')
print(conf_matrix)


print("desea guardar?")
option=int(input())

if option!=0:
    print("ingrese nombre del archivo")
    name=input()
    joblib.dump(net,f"joblib_objects\\{name}.joblib")

