import cv2
import numpy as np
import pywt
import json


img = cv2.imread('prueba.jpg')


resized_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
"""
# Redimensionar la imagen (opcional para simplificar)
resized_img = cv2.resize(img_rgb, (200, 200))
"""

coeffs = pywt.dwt2(resized_img, 'haar')  
cA, (cH, cV, cD) = coeffs

compressed_img = pywt.idwt2((cA, (None, None, None)), 'haar')
compressed_img = np.clip(compressed_img, 0, 255).astype(np.uint8)

cv2.imwrite('imagen_comprimida.jpg', cv2.cvtColor(compressed_img, cv2.COLOR_RGB2BGR))

gray = cv2.cvtColor(compressed_img, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

polygons_data = []
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [approx], -1, 255, -1)
    mean_color = cv2.mean(resized_img, mask=mask)[:3]

    polygons_data.append({
        'color': list(mean_color),
        'polygon': approx.reshape(-1, 2).tolist()
    })

with open('imagen_poligonos.json', 'w') as f:
    json.dump(polygons_data, f)


reconstructed_img = np.zeros_like(resized_img)
for poly_data in polygons_data:
    color = tuple(map(int, poly_data['color']))
    polygon = np.array(poly_data['polygon'], dtype=np.int32)
    cv2.fillPoly(reconstructed_img, [polygon], color)


cv2.imwrite('imagen_reconstruida.jpg', cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR))
