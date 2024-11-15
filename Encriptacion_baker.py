import cv2
import numpy as np

# Entrada de la longitud y los valores de la clave
longitud = int(input("Ingrese la longitud de la clave: "))
n = []
for i in range(longitud):
    valor = int(input(f"Ingrese el valor {i + 1} de la clave: "))
    n.append(valor)

# Cargar la imagen original
image = cv2.imread("burbuja.png")
N = image.shape[0]  # Asumimos que la imagen es de tamaño NxN (200x200 en este caso)

# Crear una nueva imagen vacía para almacenar los píxeles reordenados
new_image = np.zeros_like(image)

# Reordenar los píxeles según el mapa de Baker
Ni = 0
for i in range(len(n)):
    Ni = sum(n[:i])  # Calculamos Ni acumulado
    ni = n[i]  # Obtenemos el valor n_i correspondiente a la clave
    
    for x in range(N):
        for y in range(N):
            # Aplicamos la fórmula para las nuevas posiciones (x', y')
            new_x = int(((N / ni)*(x - Ni)) + (y % (N / ni)))
            new_y = int((ni / N) * (y - (y % (N / ni))) + Ni)

            # Asegúrate de que las nuevas posiciones estén dentro del rango de la imagen
            new_x = new_x % N  # Restringimos new_x dentro del rango 0..N-1
            new_y = new_y % N  # Restringimos new_y dentro del rango 0..N-1

            # Asignamos el valor del píxel original a la nueva posición en new_image
            new_image[new_x, new_y] = image[x, y]

# Guardamos la imagen reordenada
cv2.imwrite("imagen_reordenada.png", new_image)
print("La clave ingresada es:", n)