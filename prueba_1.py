import pywt        # Librería para trabajar con transformadas wavelet
import cv2         # OpenCV para cargar y guardar imágenes
import numpy as np # Librería para operaciones numéricas con matrices

# Función para reducir la resolución de la imagen y obtener los valores de reconstrucción y color
def bajar_resolucion(img_path):
    # Cargar la imagen en color (BGR)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta especificada: {img_path}")
    
    # Convertir la imagen a escala de grises para aplicar la Transformada Wavelet
    img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar la Transformada Wavelet Discreta (DWT) en escala de grises
    coeficientes = pywt.wavedec2(img_gris, 'haar', level=1)
    subimagen_baja_res = coeficientes[0]  # Subimagen en baja resolución
    detalles = coeficientes[1:]           # Coeficientes de detalle
    
    # Guardar la subimagen de baja resolución en escala de grises
    cv2.imwrite("imagen_baja_resolucion.jpg", np.clip(subimagen_baja_res, 0, 255).astype(np.uint8))
    
    # Mostrar información
    print("Subimagen Baja Resolución guardada como 'imagen_baja_resolucion.jpg'")
    
    # Retornar también los canales de color originales para la reconstrucción
    return subimagen_baja_res, detalles, img

# Función para restaurar la imagen en alta resolución y devolver el color original
def subir_resolucion(subimagen_baja_res, detalles, color_img):
    # Combinar la subimagen de baja resolución con los coeficientes de detalle para la reconstrucción
    coeficientes = [subimagen_baja_res] + detalles
    img_reconstruida_gris = pywt.waverec2(coeficientes, 'haar')
    img_reconstruida_gris = np.clip(img_reconstruida_gris, 0, 255).astype(np.uint8)
    
    # Redimensionar la imagen reconstruida en escala de grises al tamaño de la imagen original
    img_reconstruida_gris = cv2.resize(img_reconstruida_gris, (color_img.shape[1], color_img.shape[0]))

    # Aplicar el color de la imagen original a la reconstruida usando la imagen en color original
    img_reconstruida_color = cv2.merge([
        (img_reconstruida_gris * 0.9).astype(np.uint8),   # Canal azul
        (img_reconstruida_gris * 1.1).astype(np.uint8),   # Canal verde
        img_reconstruida_gris.astype(np.uint8)            # Canal rojo
    ])

    # Guardar la imagen reconstruida en color
    cv2.imwrite("imagen_reconstruida_color.jpg", img_reconstruida_color)
    
    print("Imagen reconstruida con color guardada como 'imagen_reconstruida_color.jpg'")
    return img_reconstruida_color

# Ejemplo de uso del código
imagen = 'prueva.jpg'  # Asegúrate de que esta ruta es correcta y el archivo existe
subimagen_baja_res, detalles, color_img = bajar_resolucion(imagen)
subir_resolucion(subimagen_baja_res, detalles, color_img)
