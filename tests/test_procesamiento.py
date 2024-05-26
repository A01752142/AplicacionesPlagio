# Autores: A01752142 - Sandra Ximena Téllez Olvera
#          A01749164 - Jeovani Hernandez Bastida
#          A01025261 - Maximiliano Carrasco Rojas

import unittest
from preprocesamiento import eliminar_puntuacion, procesar_parrafos, generar_ngrams, leer_documentos, similitud_coseno, calcular_distancia
import os

class TestPreprocesamiento(unittest.TestCase):
    """
    Clase de pruebas unitarias para las funciones del módulo preprocesamiento.
    """
    
    def test_eliminar_puntuacion(self):
        """
        Prueba que la función eliminar_puntuacion elimina correctamente los signos de puntuación.
        """
        texto = "Hello! How are you?"
        esperado = "Hello How are you"
        resultado = eliminar_puntuacion(texto)
        self.assertEqual(resultado, esperado)
    
    def test_procesar_parrafos(self):
        """
        Prueba que la función procesar_parrafos realiza el preprocesamiento correcto,
        incluyendo la eliminación de puntuación, tokenización, eliminación de stopwords y stemming.
        """
        texto = "Hello! How are you? I'm fine, thank you."
        esperado = ['hello', 'im', 'fine', 'thank']
        resultado = procesar_parrafos(texto)
        self.assertEqual(resultado, esperado)

    def test_generar_ngrams(self):
        """
        Prueba que la función generar_ngrams genera correctamente los n-gramas.
        """
        palabras = ['hello', 'fine', 'thank']
        esperado = ['hello fine', 'fine thank']
        resultado = generar_ngrams(palabras, 2)
        self.assertEqual(resultado, esperado)

    def test_leer_documentos(self):
        """
        Prueba que la función leer_documentos lee correctamente los documentos de una carpeta.
        Nota: Asume que hay documentos en la carpeta 'Documentos'.
        """
        # Crear una carpeta temporal con archivos de prueba
        os.makedirs('Documentos_Test', exist_ok=True)
        with open('Documentos_Test/test1.txt', 'w', encoding='utf-8') as f:
            f.write("This is a test document.")
        
        documentos = leer_documentos('Documentos_Test')
        self.assertGreater(len(documentos), 0)

        # Eliminar la carpeta temporal y sus archivos
        os.remove('Documentos_Test/test1.txt')
        os.rmdir('Documentos_Test')

    def test_similitud_coseno(self):
        """
        Prueba que la función similitud_coseno calcula correctamente la similitud del coseno entre dos vectores.
        """
        v1 = [1, 2, 3]
        v2 = [1, 2, 3]
        esperado = 1.0
        resultado = similitud_coseno(v1, v2)
        self.assertAlmostEqual(resultado, esperado, places=5)

    def test_calcular_distancia(self):
        """
        Prueba que la función calcular_distancia calcula correctamente la similitud del coseno
        entre los n-gramas de dos listas de palabras.
        """
        p1 = ['hello', 'fine', 'thank']
        p2 = ['hello', 'thank']
        resultado = calcular_distancia(p1, p2, 2)
        # Verificamos que el resultado esté en el rango esperado
        self.assertTrue(0.0 <= resultado <= 1.0)

if __name__ == '__main__':
    unittest.main()
