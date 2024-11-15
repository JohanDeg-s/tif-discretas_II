import cv2
import numpy as np

longitud = int(input("Ingrese la longitud de la clave: "))
n = []
for i in range(longitud):
    valor = int(input(f"Ingrese el valor {i + 1} de la clave: "))
    n.append(valor)


image = cv2.imread("burbuja.png")
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