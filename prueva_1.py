import pywt
import cv2
import numpy as np

# Función para reducir la resolución de la imagen y obtener los valores de reconstrucción
def bajar_resolucion(img_path):
    # Cargar imagen en escala de grises
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Aplicar la Transformada Wavelet Discreta (DWT)
    coeficientes = pywt.wavedec2(img, 'haar', level=1)
    # Separar la subimagen de baja resolución y los coeficientes de detalle
    subimagen_baja_res = coeficientes[0]
    detalles = coeficientes[1:]
    
    # Mostrar la subimagen y los coeficientes necesarios para la reconstrucción
    print("Subimagen Baja Resolución:\n", subimagen_baja_res)
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
    
    # Mostrar la imagen reconstruida
    cv2.imshow("Imagen Reconstruida", img_reconstruida)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return img_reconstruida
