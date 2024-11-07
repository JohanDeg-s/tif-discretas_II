import pywt        # Librería para trabajar con transformadas wavelet
import cv2         # OpenCV para cargar y guardar imágenes
import numpy as np # Librería para operaciones numéricas con matrices

# Función para reducir la resolución de la imagen y obtener los valores de reconstrucción
def bajar_resolucion(img_path):
    # Cargar imagen en escala de grises
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en la ruta especificada: {img_path}")
    
    # Aplicar la Transformada Wavelet Discreta (DWT)
    coeficientes = pywt.wavedec2(img, 'haar', level=1)
    # Separar la subimagen de baja resolución y los coeficientes de detalle
    subimagen_baja_res = coeficientes[0]
    detalles = coeficientes[1:]
    
    # Guardar la subimagen de baja resolución como archivo JPEG sin cambiar su tamaño
    cv2.imwrite("imagen_baja_resolucion.jpg", np.clip(subimagen_baja_res, 0, 255).astype(np.uint8))
    
    # Mostrar la subimagen y los coeficientes necesarios para la reconstrucción
    print("Subimagen Baja Resolución guardada como 'imagen_baja_resolucion.jpg'")
    print("\nCoeficientes de Detalle:")
    for i, detalle in enumerate(detalles, start=1):
        print(f"Detalle Nivel {i} (Horizontal, Vertical, Diagonal):\n", detalle)
    
    return subimagen_baja_res, detalles

# Función para restaurar la imagen en alta resolución utilizando los valores de reconstrucción
def subir_resolucion(subimagen_baja_res, detalles):
    # Combinar la subimagen de baja resolución con los coeficientes de detalle
    coeficientes = [subimagen_baja_res] + detalles
    # Aplicar la Transformada Inversa Wavelet (IDWT) para restaurar la imagen
    img_reconstruida = pywt.waverec2(coeficientes, 'haar')
    img_reconstruida = np.clip(img_reconstruida, 0, 255).astype(np.uint8)
    
    # Guardar la imagen reconstruida en un archivo
    cv2.imwrite("imagen_reconstruida.jpg", img_reconstruida)
    
    print("Imagen reconstruida guardada como 'imagen_reconstruida.jpg'")
    return img_reconstruida

# Ejemplo de uso del código
imagen = 'prueba.jpg'  # Asegúrate de que esta ruta es correcta y el archivo existe
subimagen_baja_res, detalles = bajar_resolucion(imagen)
subir_resolucion(subimagen_baja_res, detalles)
