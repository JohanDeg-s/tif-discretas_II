import numpy as np
from scipy.fft import fft2, fftshift
import imageio  #imagenes

ruta = 'a.jpg'

grises = imageio.imread(ruta, mode='L') #escala grises

# Fourier - fila
f_transform_rows = np.array([np.fft.fft(row) for row in ruta]) 

# Fourier - columna
f_transform = np.array([np.fft.fft(col) for col in f_transform_rows.T]).T 

# Valor abs
e = np.log(np.abs(f_transform) + 1)  

# Guardar el espectrograma como archivo numpy
np.save('e.npy', e)
print("Espectrograma - npy")

# Guardar el espectrograma como imagen PNG
imageio.imwrite('e.png', np.uint8(e / np.max(e) * 255))  # Normalización de la imagen
print("Espectrograma - png")
