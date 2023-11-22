#script para extraer de un maldito csv los correos que no son spam y los que si son
#que grandes por andar usando csv si algo dificil tenia que tener esta cosa
#might delete later
import pandas as pd
import os

# Cargar el archivo CSV
archivo_csv = "C:\\Users\\LENOVO\\Desktop\\sem IA 2\\proyecto_final\\spam_ham_dataset.csv" 
# Reemplaza con la ruta real de tu archivo
datos = pd.read_csv(archivo_csv)

# Crear directorios para spam y no_spam
directorio_spam = "C:\\Users\\LENOVO\\Desktop\\sem IA 2\\proyecto_final\\spam"
directorio_no_spam = "C:\\Users\\LENOVO\\Desktop\\sem IA 2\\proyecto_final\\easy_ham"

# Iterar sobre los datos y clasificar correos
for indice, fila in datos.iterrows():
    correo = fila['text']
    label = fila['label_num']

    if label == 1:
        # Spam
        with open(os.path.join(directorio_spam, f"spam_{indice}_test.txt"), "w",encoding="utf-8") as archivo:
            archivo.write(correo)
    else:
        # No Spam
        with open(os.path.join(directorio_no_spam, f"no_spam_{indice}_test.txt"), "w",encoding="utf-8") as archivo:
            archivo.write(correo)

print("Proceso completado. Correos clasificados y guardados en directorios.")