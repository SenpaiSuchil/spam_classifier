import os

# Directorio que contiene los archivos que deseas renombrar
directorio = "proyecto_final\\easy_ham\\"

# Nueva extensión deseada
nueva_extension = ".txt"

# Recorre los archivos en el directorio
for nombre_archivo in os.listdir(directorio):
    archivo_actual = os.path.join(directorio, nombre_archivo)
    
    # Verifica si el archivo es un archivo regular
    if os.path.isfile(archivo_actual):
        # Obtiene el nombre del archivo sin la extensión original
        nombre_base, extension_original = os.path.splitext(nombre_archivo)
        
        # Renombra el archivo con la nueva extensión .txt
        nuevo_nombre = nombre_base + nueva_extension
        archivo_nuevo = os.path.join(directorio, nuevo_nombre)
        
        # Realiza la operación de renombrar
        os.rename(archivo_actual, archivo_nuevo)

print("Archivos renombrados a .txt")