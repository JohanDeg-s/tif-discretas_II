#Imagen -> espectograma

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def generar_espectrogramas(imagen_path, output_prefix):
    imagen = Image.open(imagen_path).convert("RGB")
    imagen_array = np.array(imagen)

    canal_rojo = imagen_array[:, :, 0]
    canal_verde = imagen_array[:, :, 1]
    canal_azul = imagen_array[:, :, 2]

    espectro_rojo = np.fft.fftshift(np.fft.fft2(canal_rojo))
    espectro_verde = np.fft.fftshift(np.fft.fft2(canal_verde))
    espectro_azul = np.fft.fftshift(np.fft.fft2(canal_azul))

    np.save(f"{output_prefix}_rojo.npy", espectro_rojo)
    np.save(f"{output_prefix}_verde.npy", espectro_verde)
    np.save(f"{output_prefix}_azul.npy", espectro_azul)

    for espectro, color, nombre in zip(
        [espectro_rojo, espectro_verde, espectro_azul],
        ['Reds', 'Greens', 'Blues'],
        ['Rojo', 'Verde', 'Azul']
    ):
        plt.figure(figsize=(6, 6))
        plt.imshow(np.log1p(np.abs(espectro)), cmap=color)
        plt.axis('off')
        plt.title(f"Espectrograma ({nombre})")
        plt.show()

generar_espectrogramas("a.jpg", "x1-espectrograma")













# Espectograma -> audio
import numpy as np
import wave
import os

def espectrograma_a_audio(espectrograma_path, audio_output_path, sample_rate=44100):
    espectrograma = np.load(espectrograma_path)

    espectrograma = np.fft.ifftshift(espectrograma)

    señal_audio = np.fft.ifft(espectrograma).real

    señal_audio -= señal_audio.min()
    señal_audio /= señal_audio.max()
    señal_audio = 2 * señal_audio - 1

    señal_audio = (señal_audio * 32767).astype(np.int16)

    with wave.open(audio_output_path, 'w') as archivo_wav:
        # Configurar parámetros del archivo WAV
        archivo_wav.setnchannels(1)  # Mono
        archivo_wav.setsampwidth(2)  # 2 bytes (16 bits)
        archivo_wav.setframerate(sample_rate)  # Frecuencia de muestreo
        archivo_wav.writeframes(señal_audio.tobytes())

    print(f"Audio reconstruido guardado en: {audio_output_path}")

# Convertir canal rojo a audio
espectrograma_a_audio(
    espectrograma_path='x1-espectrograma_azul.npy', 
    audio_output_path='x2-audio_rojo.wav',      
    sample_rate=44100                           
)

espectrograma_a_audio(
    espectrograma_path='x1-espectrograma_verde.npy',
    audio_output_path='x2-audio_verde.wav',          
    sample_rate=44100                         
)

espectrograma_a_audio(
    espectrograma_path='x1-espectrograma_azul.npy',  
    audio_output_path='x2-audio_azul.wav',          
    sample_rate=44100                            
)
















# Unir audios RGB a uno solo
from pydub import AudioSegment

audio_rojo = AudioSegment.from_wav  ("x2-audio_rojo.wav")
audio_verde = AudioSegment.from_wav ("x2-audio_verde.wav")
audio_azul = AudioSegment.from_wav  ("x2-audio_azul.wav")

print("Duración del audio rojo:", audio_rojo.duration_seconds, "segundos")
print("Duración del audio verde:", audio_verde.duration_seconds, "segundos")
print("Duración del audio azul:", audio_azul.duration_seconds, "segundos")

audio_combinado = audio_rojo + audio_verde + audio_azul

audio_combinado.export("x3-audio_combinado.wav", format="wav")

audio_combinado = AudioSegment.from_wav("x3-audio_combinado.wav")
print("Duración total del audio combinado:", audio_combinado.duration_seconds, "segundos")



