import numpy as np
import cv2

def baker_map_inverse(xn, yn, p):
    """Aplica el mapa de Baker de forma inversa"""
    if 0 < xn <= p:
        xn_inv = xn * p
        yn_inv = yn / p
    else:
        xn_inv = (xn - p) / (1 - p)
        yn_inv = (yn - (1 - p)) / (1 - p)
    return xn_inv, yn_inv

def logistic_map_inverse(zn, mu):
    """Aplica el mapa logístico de forma inversa"""
    for _ in range(50):  # iteraciones para aproximar la inversa
        zn_inv = zn / mu - (1 - zn) / mu
    return zn_inv

def reverse_image_encryption(encrypted_image_path, p, mu):
    # Cargar la imagen cifrada
    encrypted_image = cv2.imread(encrypted_image_path)
    decrypted_image = np.zeros_like(encrypted_image)

    # Obtener las dimensiones de la imagen
    height, width, channels = encrypted_image.shape
    
    # Recorrer todos los píxeles y aplicar las transformaciones inversas
    for i in range(height):
        for j in range(width):
            # Obtener los valores RGB del píxel cifrado
            r, g, b = encrypted_image[i, j]
            
            # Aplicar la transformación inversa para cada canal RGB
            xn = r / 255.0
            yn = g / 255.0
            zn = b / 255.0
            
            # Aplicar el mapa de Baker inverso
            xn_inv, yn_inv = baker_map_inverse(xn, yn, p)
            
            # Aplicar el mapa logístico inverso
            zn_inv = logistic_map_inverse(zn, mu)
            
            # Recuperar los valores originales multiplicándolos por 255
            r_inv = int(np.clip(xn_inv * 255, 0, 255))  # Usar np.clip para asegurar el rango
            g_inv = int(np.clip(yn_inv * 255, 0, 255))  # Usar np.clip para asegurar el rango
            b_inv = int(np.clip(zn_inv * 255, 0, 255))  # Usar np.clip para asegurar el rango

            # Asignar los valores al píxel desencriptado
            decrypted_image[i, j] = [r_inv, g_inv, b_inv]

    # Guardar la imagen decodificada
    cv2.imwrite("reversed_image.png", decrypted_image)

# Parámetros del mapa de caos
p = 0.5  # Puedes ajustar este valor según lo que hayas usado en la encriptación
mu = 3.9  # Puedes ajustar este valor también

# Llamar a la función con la imagen cifrada
reverse_image_encryption("imagen_baja_resolucion.jpg", p, mu)
