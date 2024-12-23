import cv2
import numpy as np

def bajar_resolucion(img_path, target_size=(200, 200)):
    img = cv2.imread(img_path)
    
    original_shape = img.shape

    img_bn = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_redimensionada = cv2.resize(img_bn, target_size)

    cv2.imwrite("imagen_redimensionada_bn.jpg", img_redimensionada)
    
    return img_redimensionada, original_shape

def subir_resolucion(img_redimensionada, original_shape):

    img_restaurada = cv2.resize(img_redimensionada, (original_shape[1], original_shape[0]))
    
    cv2.imwrite("imagen_restaurada.jpg", img_restaurada)
    
    print("Imagen restaurada al tamaño original guardada como 'imagen_restaurada.jpg'")
    
    return img_restaurada

imagen = 'prueba.jpg'
# Redimensionar a 200x200 píxeles en blanco y negro
img_redimensionada, original_shape = bajar_resolucion(imagen, target_size=(200, 200))

subir_resolucion(img_redimensionada, original_shape)
