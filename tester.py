import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from MLP_batch import *
import random
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from keras.preprocessing.sequence import pad_sequences

spam_test = "spam_test\\"
not_spam_test = "not_spam_test\\"


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
    #random.shuffle(emails)
    

    #este ciclo extrae los correos de la posicion 0 del arreglo emails para almacenarlo en la variable correos
    correos = [correo[0] for correo in emails]
    #lo mismo pero en la pos 1 del arreglo para extraer su etiqueta
    etiquetas = np.array([[correo[1] for correo in emails]])

    return correos,etiquetas


print("ingrese el nombre del archivo (sin extension)")
name=input()

net=joblib.load(f"joblib_objects\\{name}.joblib")

spam =mail_extractor(spam_test,400)
not_spam=mail_extractor(not_spam_test,400)

mails,labels=mail_mixer(spam, not_spam)

vectorizer = CountVectorizer()
# Aplicar BoW a los correos electrónicos
X_bow =vectorizer.fit_transform(mails)

#transformamos a arreglo
#(no lo mas optimo pero la matriz disperza por alguna extraña razon no me dejaba operar)
X=X_bow.toarray()


X_padded = pad_sequences(X, maxlen=30000, padding='post', truncating='post')
minimo = np.min(X_padded)
rango = np.max(X_padded) - minimo

# Aplicar la normalización Min-Max
X_padded = (X_padded - minimo) / rango

X_padded=X_padded.T

y_pred_log=net.predict(X_padded)

umbral = 0.5

# Etiquetado binario utilizando el umbral
y_pred = (y_pred_log >= umbral).astype(int)

# Calcular precisión (accuracy)
accuracy = accuracy_score(labels[0], y_pred[0])
print(f'Accuracy: {accuracy:.4f}')

# Calcular puntaje F1
f1 = f1_score(labels[0], y_pred[0])
print(f'F1 Score: {f1:.4f}')

# Calcular matriz de confusión
conf_matrix = confusion_matrix(labels[0], y_pred[0])
print('Matriz de Confusión:')
print(conf_matrix)


