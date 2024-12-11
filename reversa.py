# Separar audios

from pydub import AudioSegment
audio_combinado = AudioSegment.from_wav("x3-audio_combinado.wav")
duracion_canal_ms = 1417.233

audio_rojo = audio_combinado[:duracion_canal_ms]
audio_verde = audio_combinado[duracion_canal_ms:2*duracion_canal_ms]
audio_azul = audio_combinado[2*duracion_canal_ms:3*duracion_canal_ms]

audio_rojo.export ("z1-audio_rojo_separado.wav"  , format="wav")
audio_verde.export("z1-audio_verde_separado.wav" , format="wav")
audio_azul.export ("z1-audio_azul_separado.wav"  , format="wav")

print("Audios separados y exportados correctamente.")









# Audio -> Espectograma

import numpy as np
import wave
import matplotlib.pyplot as plt

def audio_a_espectrograma(audio_path, output_prefix, sample_rate=44100, window_size=1024, overlap=512, color='viridis'):
    with wave.open(audio_path, 'r') as audio_file:
        n_channels = audio_file.getnchannels()
        sample_width = audio_file.getsampwidth()
        n_frames = audio_file.getnframes()
        framerate = audio_file.getframerate()

        if framerate != sample_rate:
            print(f"Advertencia: Frecuencia de muestreo esperada ({sample_rate}) "
                  f"no coincide con la del archivo ({framerate}).")
        
        audio_frames = audio_file.readframes(n_frames)
        dtype = np.int16 if sample_width == 2 else np.int8
        audio_data = np.frombuffer(audio_frames, dtype=dtype)

        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels)
            audio_data = audio_data[:, 0]
        
        espectrograma = []

        for start in range(0, len(audio_data) - window_size, window_size - overlap):
            ventana = audio_data[start:start + window_size]
            espectro = np.fft.fft(ventana)
            espectrograma.append(np.abs(espectro))

        espectrograma = np.array(espectrograma).T

        espectrograma_path = f"{output_prefix}.npy"
        np.save(espectrograma_path, espectrograma)
        print(f"Espectrograma guardado en: {espectrograma_path}")
        
        plt.figure(figsize=(10, 6))
        
        if color == 'blue':
            cmap = 'Blues'
        elif color == 'red':
            cmap = 'Reds'
        elif color == 'green':
            cmap = 'Greens'
        else:
            cmap = 'viridis'

        plt.imshow(np.log1p(espectrograma), aspect='auto', cmap=cmap, origin='lower')
        plt.axis('off')  # Eliminar los ejes y etiquetas
        imagen_path = f"{output_prefix}.png"
        plt.savefig(imagen_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Imagen del espectrograma guardada en: {imagen_path}")

audio_a_espectrograma(
    audio_path='z1-audio_azul_separado.wav',  # Ruta al archivo de audio
    output_prefix='z2-espectrograma_azul',       # Prefijo de salida
    sample_rate=44100,                        # Frecuencia de muestreo
    color='blue'                              # Color del espectrograma
)

audio_a_espectrograma(
    audio_path='z1-audio_rojo_separado.wav',  # Ruta al archivo de audio
    output_prefix='z2-espectrograma_rojo',       # Prefijo de salida
    sample_rate=44100,                        # Frecuencia de muestreo
    color='red'                               # Color del espectrograma
)

audio_a_espectrograma(
    audio_path='z1-audio_verde_separado.wav', # Ruta al archivo de audio
    output_prefix='z2-espectrograma_verde',      # Prefijo de salida
    sample_rate=44100,                        # Frecuencia de muestreo
    color='green'                             # Color del espectrograma
)










# Espectograma -> Imagen
#funciona
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import spectrogram
from PIL import Image

def generar_espectrograma_audio(audio_ruta, color="R", tamaño=250):
    sr, audio = read(audio_ruta)

    if len(audio.shape) > 1:  # Si el audio es estéreo, se toma solo el primer canal
        audio = audio[:, 0]

    frecuencias, tiempos, Sxx = spectrogram(audio, fs=sr, nperseg=1024, noverlap=512)

    espectro = np.log(1 + Sxx)

    espectro_redimensionado = np.resize(espectro, (tamaño, tamaño))

    espectro_shifted = np.fft.fftshift(espectro_redimensionado)

    if color == "R":
        np.save(f"z2-espectrograma_rojo.npy", espectro_shifted)
    elif color == "G":
        np.save(f"z2-espectrograma_verde.npy", espectro_shifted)
    elif color == "B":
        np.save(f"z2-espectrograma_azul.npy", espectro_shifted)

    return espectro_redimensionado

def main():
    ruta_audio_rojo  = "z1-audio_rojo_separado.wav"  
    ruta_audio_verde = "z1-audio_verde_separado.wav"  
    ruta_audio_azul  = "z1-audio_azul_separado.wav"  

    generar_espectrograma_audio(ruta_audio_rojo, color="R")
    generar_espectrograma_audio(ruta_audio_verde, color="G")
    generar_espectrograma_audio(ruta_audio_azul, color="B")

if __name__ == "__main__":
    main()







