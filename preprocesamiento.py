# Autores: A01752142 - Sandra Ximena Téllez Olvera
#          A01749164 - Jeovani Hernandez Bastida
#          A01025261 - Maximiliano Carrasco Rojas

import os
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from collections import Counter
import numpy as np

# Asegurarse de que los recursos de NLTK estén descargados
nltk.download('punkt')
nltk.download('stopwords')

# Inicializar el stemmer de Snowball para inglés
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))

def eliminar_puntuacion(texto):
    """
    Función que elimina la puntuación del texto dado.
    Args: 
        texto (str): El texto del cual eliminar la puntuación.
    Returns:
        str: El texto sin puntuación.
    """
    
    # Crear una tabla de traducción que mapea cada signo de puntuación a None
    tabla_traduccion = str.maketrans('', '', string.punctuation)
    
    # Eliminar los signos de puntuación usando la tabla de traducción
    texto_sin_puntuacion = texto.translate(tabla_traduccion)
    
    return texto_sin_puntuacion

def procesar_parrafos(text):
    """
    Función que realiza el preprocesamiento del texto dado.
    Args:
        text (str): El texto a procesar.
    Returns:
        list: Lista de palabras procesadas.
    """
    texto = text.lower()                                                              # Estandarizamos a minúsculas
    texto = re.sub(r'[^\w\s]', '', texto)                                             # Eliminamos la puntuación
    palabras = word_tokenize(texto)                                                   # Dividimos los párrafos en palabras
    palabras = [stemmer.stem(word) for word in palabras if word not in stop_words]    # Nos quedamos con la raíz de la palabra y aplicamos stemming
    return palabras

def generar_ngrams(words, n):
    """
    Función que genera n-gramas a partir de una lista de palabras.
    Args:
        words (list): Lista de palabras.
        n (int): Longitud del n-grama.
    Returns:
        list: Lista de n-gramas generados.
    """
    ngrams = zip(*[words[i:] for i in range(n)])
    return [' '.join(ngram) for ngram in ngrams]

def leer_documentos(carpeta):
    """
    Función que lee documentos de la carpeta especificada y los procesa.
    Args:
        carpeta (str): Nombre de la carpeta que contiene los documentos.
    Returns:
        dict: Diccionario con nombres de archivos como claves y listas de palabras procesadas como valores.
    """
    
    documentos_path = os.path.join(os.path.dirname(__file__), carpeta)  # Ruta
    documentos = {}  # Diccionario para almacenar los documentos
    
    # Iterar sobre todos los archivos en Documentos
    for nombre_archivo in os.listdir(documentos_path):
        ruta_archivo = os.path.join(documentos_path, nombre_archivo)
        
        # Verificar que sea un archivo y no una subcarpeta
        if os.path.isfile(ruta_archivo):
            with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
                contenido = archivo.read()
                contenido_procesado = procesar_parrafos(contenido)
                documentos[nombre_archivo] = contenido_procesado
                
    return documentos

def similitud_coseno(v1, v2):
    """
    Función que calcula la similitud del coseno entre dos vectores.
    Args:
        v1 (array): Primer vector.
        v2 (array): Segundo vector.
    Returns:
        float: Similitud del coseno entre los dos vectores.
    """
    
    producto_punto = np.dot(v1, v2)
    vec1 = np.linalg.norm(v1)
    vec2 = np.linalg.norm(v2)
    return producto_punto / (vec1 * vec2)

def calcular_distancia(p1, p2, n):
    """
    Función que calcula la distancia entre dos documentos usando n-gramas.    
    Args:
        p1 (list): Lista de palabras del primer documento.
        p2 (list): Lista de palabras del segundo documento.
        n (int): Longitud del n-grama.
    Returns:
        float: Similitud del coseno entre los vectores de frecuencia de los n-gramas de los documentos.
    """
    
    # Generamos n-gramas
    ngrams1 = generar_ngrams(p1, n)
    ngrams2 = generar_ngrams(p2, n)

    # Contamos la frecuencia de los n-gramas
    c1 = Counter(ngrams1)
    c2 = Counter(ngrams2)

    # Creamos un conjunto de todos los n-gramas
    ngramsT = set(c1.keys()).union(set(c2.keys()))

    # Creamos vectores de frecuencia
    vector1 = np.array([c1[ngram] for ngram in ngramsT])
    vector2 = np.array([c2[ngram] for ngram in ngramsT])

    # Calculamos la similitud del coseno
    similitud = similitud_coseno(vector1, vector2)

    return similitud

# Leer documentos de ambas carpetas
documentos = leer_documentos('Documentos')

# Listar archivos disponibles en Documentos_Comparar
print("Archivos disponibles para analizar:")
archivos_comparar = os.listdir('Documentos_Comparar')
for i, nombre_archivo in enumerate(archivos_comparar):
    print(f"{i + 1}. {nombre_archivo}")

# Solicitar al usuario que elija un archivo para comparar
opcion = int(input("Seleccione el número del archivo que desea comparar: ")) - 1
if opcion < 0 or opcion >= len(archivos_comparar):
    print("Opción inválida.")
else:
    nombre_archivo_comparar = archivos_comparar[opcion]
    documento_comparar = leer_documentos('Documentos_Comparar')[nombre_archivo_comparar]

    # Variable de control para detectar si hay al menos un caso de plagio o reutilización de texto
    caso_detectado = False

    # Aplicar la función de distancia entre el documento seleccionado y cada documento de Documentos
    print(f"\nComparando el archivo '{nombre_archivo_comparar}' con documentos en la carpeta 'Documentos':")
    for nombre_documento, contenido_documento in documentos.items():
        distancia = calcular_distancia(documento_comparar, contenido_documento, 2)
        if distancia > 0.10:
            print(f"  {nombre_documento}: {(distancia * 100):.2f}% - Plagio detectado")
            caso_detectado = True
        elif 0.05 < distancia <= 0.10:
            print(f"  {nombre_documento}: {(distancia * 100):.2f}% - Reutilización de texto detectada")
            caso_detectado = True
    print()

    # Si no se ha detectado ningún caso de plagio o reutilización de texto, imprimir "Archivo libre de plagio"
    if not caso_detectado:
        print("Archivo libre de plagio")