import cv2
import numpy as np

#redimencionamos y ponemos a blanco y negro funcion
def bajar_resolucion(img_path, target_size=(200, 200)):
    img = cv2.imread(img_path)
    
    original_shape = img.shape

    img_bn = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_redimensionada = cv2.resize(img_bn, target_size)

    cv2.imwrite("imagen_redimensionada_bn.jpg", img_redimensionada)
    
    return img_redimensionada, original_shape




imagen = 'fondo_hd.jpg'
# Redimensionar a 200x200 píxeles en blanco y negro
img_redimensionada, original_shape = bajar_resolucion(imagen, target_size=(200, 200))

#empesamos con la emcriptacion

longitud = int(input("Ingrese la longitud de la clave: "))
n = []
for i in range(longitud):
    valor = int(input(f"Ingrese el valor {i + 1} de la clave: "))
    n.append(valor)


image = cv2.imread("imagen_redimensionada_bn.jpg")
N = image.shape[0]  

new_image = np.zeros_like(image)

# Reordenar los píxeles según el mapa de Baker
Ni = 0
for i in range(len(n)):
    Ni = sum(n[:i])  
    ni = n[i] 
    
    for x in range(N):
        for y in range(N):
            new_x = int(((N / ni)*(x - Ni)) + (y % (N / ni)))
            new_y = int((ni / N) * (y - (y % (N / ni))) + Ni)

            # Asegúrate de que las nuevas posiciones estén dentro del rango de la imagen
            new_x = new_x % N  # Restringimos new_x dentro del rango 0..N-1
            new_y = new_y % N  # Restringimos new_y dentro del rango 0..N-1


            new_image[new_x, new_y] = image[x, y]

cv2.imwrite("imagen_reordenada.png", new_image)
print("La clave ingresada es:", n)