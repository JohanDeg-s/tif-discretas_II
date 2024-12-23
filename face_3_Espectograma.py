import numpy as np
from PIL import Image
import librosa
import soundfile as sf

def imagen_a_espectrograma(ruta_imagen, nombre_base, n_fft=2048, hop_length=512):
    try:
        imagen = Image.open(ruta_imagen).convert('L')
        imagen = np.array(imagen, dtype=np.float32)

        altura_original, ancho_original = imagen.shape
        imagen /= 255.0

        audio = imagen.flatten()
        espectrograma = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitud = np.abs(espectrograma)
        fase = np.angle(espectrograma)
        np.save(f'a-{nombre_base}_magnitud.npy', magnitud)
        np.save(f'a-{nombre_base}_fase.npy', fase)
        np.save(f'a-{nombre_base}_dimensiones.npy', (altura_original, ancho_original)) 
        print(f'Magnitud y fase guardados para a-{nombre_base}.')

        espectrograma_reconstruido = magnitud * np.exp(1j * fase)
        audio_reconstruido = librosa.istft(espectrograma_reconstruido)

        audio_reconstruido /= np.max(np.abs(audio_reconstruido))

        sf.write(f'audio_de_{nombre_base}.wav', audio_reconstruido, 44100)
        print(f'Audio guardado como audio_de_{nombre_base}.wav.')

    except Exception as e:
        print(f"Error al procesar {ruta_imagen}: {e}")



ruta_imagen = '2.jpg'
nombre_base = 'imagen'
imagen_a_espectrograma(ruta_imagen, nombre_base)
