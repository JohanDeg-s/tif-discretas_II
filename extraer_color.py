import cv2
import numpy as np
from sklearn.cluster import KMeans

# Cargar la imagen
img = cv2.imread('imagen.jpg')

# Convertir a formato RGB (OpenCV usa BGR por defecto)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Redimensionar para reducir complejidad (opcional)
resized_img = cv2.resize(img_rgb, (200, 200))  # Reducimos la imagen para simplificar el procesamiento

# Reshape para usar k-means
pixels = resized_img.reshape((-1, 3))

# Aplicar K-means para encontrar colores base
n_clusters = 4  # Ajustar según necesidad
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(pixels)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Reconstruir la imagen segmentada
segmented_img = centers[labels].reshape(resized_img.shape).astype(np.uint8)

# Mostrar la imagen segmentada
cv2.imshow('Segmentación', segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convertir la imagen segmentada a escala de grises
segmented_gray = cv2.cvtColor(segmented_img, cv2.COLOR_RGB2GRAY)

# Detectar bordes con Canny
edges = cv2.Canny(segmented_gray, 50, 150)

# Mostrar bordes detectados
cv2.imshow('Bordes', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Encontrar contornos
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crear una imagen en blanco para dibujar los contornos
contours_img = np.zeros_like(segmented_gray)

# Dibujar los contornos aproximados
for contour in contours:
    # Aproximar el contorno con un polígono
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Dibujar el contorno aproximado
    cv2.drawContours(contours_img, [approx], -1, 255, -1)  # Color blanco (255)

cv2.imshow('Aproximación Poligonal', contours_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import json

# Crear estructura de datos para almacenar polígonos y colores
polygons_data = []
for i, contour in enumerate(contours):
    # Obtener el color base del área
    color = centers[i % len(centers)]  # Ajustar índice según el número de regiones
    
    # Almacenar las coordenadas del polígono y el color
    polygons_data.append({
        'color': color.tolist(),
        'polygon': contour.reshape(-1, 2).tolist()
    })

# Guardar en un archivo JSON
with open('imagen_poligonos.json', 'w') as f:
    json.dump(polygons_data, f)
    
# Leer los datos del archivo JSON
with open('imagen_poligonos.json', 'r') as f:
    polygons_data = json.load(f)

# Crear una imagen en blanco
reconstructed_img = np.zeros_like(segmented_img)

# Dibujar cada polígono con su color
for poly_data in polygons_data:
    color = tuple(map(int, poly_data['color']))  # Convertir color a enteros
    polygon = np.array(poly_data['polygon'], dtype=np.int32)
    cv2.fillPoly(reconstructed_img, [polygon], color)

cv2.imshow('Reconstrucción', reconstructed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

