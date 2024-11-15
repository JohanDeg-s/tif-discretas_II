import pywt        
import cv2         
import numpy as np 

def bajar_resolucion(img_path, target_size=(500, 500)):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta especificada: {img_path}")
    

    original_shape = img.shape
    
    img_redimensionada = cv2.resize(img, target_size)
    

    coeficientes = pywt.wavedec2(img_redimensionada, 'haar', level=1)
    subimagen_baja_res = coeficientes[0]
    detalles = coeficientes[1:]
    

    cv2.imwrite("imagen_baja_resolucion.jpg", np.clip(subimagen_baja_res, 0, 255).astype(np.uint8))
    
    print("Subimagen Baja Resolución guardada como 'imagen_baja_resolucion.jpg'")
    print("\nCoeficientes de Detalle:")
    for i, detalle in enumerate(detalles, start=1):
        print(f"Detalle Nivel {i} (Horizontal, Vertical, Diagonal):\n", detalle)
    
    return subimagen_baja_res, detalles, original_shape

def subir_resolucion(subimagen_baja_res, detalles, original_shape):
    
    coeficientes = [subimagen_baja_res] + detalles

    img_reconstruida = pywt.waverec2(coeficientes, 'haar')
    img_reconstruida = np.clip(img_reconstruida, 0, 255).astype(np.uint8)
    
    img_reconstruida_original = cv2.resize(img_reconstruida, (original_shape[1], original_shape[0]))
    
    cv2.imwrite("imagen_reconstruida.jpg", img_reconstruida_original)
    
    print("Imagen reconstruida guardada como 'imagen_reconstruida.jpg'")
    return img_reconstruida_original

imagen = 'prueba.jpg'  
subimagen_baja_res, detalles, original_shape = bajar_resolucion(imagen, target_size=(500, 500))
subir_resolucion(subimagen_baja_res, detalles, original_shape)
