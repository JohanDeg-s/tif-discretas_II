import imageio.v2 as imageio
import numpy as np
import math
import wave
import cmath

ruta = 'imagen_redimensionada_bn.jpg'
imagen = imageio.imread(ruta, mode='F')
imagen = imagen[::2, ::2]  

N, M = imagen.shape


def dft2d(imagen):
    N, M = imagen.shape
    F = np.zeros((N, M), dtype=complex)  
    for u in range(N):
        for v in range(M):
            suma = 0.0
            for x in range(N):
                for y in range(M):
                    angulo = -2j * math.pi * ((u * x / N) + (v * y / M))
                    suma += imagen[x, y] * cmath.exp(angulo)
            F[u, v] = suma

    return F


espectrograma_dft = dft2d(imagen)
espectrograma_log = np.log(np.abs(espectrograma_dft) + 1)

np.save('espectrograma.npy', espectrograma_log)
print("Espectrograma guardado como archivo .npy")

espectrograma_normalizado = np.uint8(espectrograma_log / np.max(espectrograma_log) * 255)
imageio.imwrite('espectrograma.png', espectrograma_normalizado)
print("Espectrograma guardado como imagen PNG")


def espectrograma_a_audio(espectrograma, seg=5, frecuencia_muestreo=44100):
    N, M = espectrograma.shape
    num_muestras = seg * frecuencia_muestreo
    tiempo_total = np.linspace(0, seg, num_muestras)
    audio = np.zeros(num_muestras)

    for v in range(M):
            frecuencia = (u + 1) * (v + 1) / (N + M)  
            angulo = 2 * math.pi * frecuencia * tiempo_total
            componente = np.real(espectrograma[u, v]) * np.cos(angulo) 
            audio += componente  

    audio = audio / np.max(np.abs(audio)) * 32767 
    audio = audio.astype(np.int16)

    return audio


audio_data = espectrograma_a_audio(espectrograma_dft, duracion_segundos=40)

with wave.open('audio2.wav', 'wb') as archivo_wav:
    archivo_wav.setnchannels(1)  
    archivo_wav.setsampwidth(2)  
    archivo_wav.setframerate(44100)  
    archivo_wav.writeframes(audio_data.tobytes())

print("Audio guardado")

