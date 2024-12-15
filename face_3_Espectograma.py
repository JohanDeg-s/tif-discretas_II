import numpy as np
from PIL import Image
import librosa
import soundfile as sf

def imagen_a_espectrograma(ruta_imagen, nombre_base, n_fft=2048, hop_length=512):
    try:
        # Cargar la imagen y convertirla a escala de grises
        imagen = Image.open(ruta_imagen).convert('L')
        imagen = np.array(imagen, dtype=np.float32)

        # Guardar las dimensiones originales
        altura_original, ancho_original = imagen.shape

        # Normalizar la imagen
        imagen /= 255.0

        # Convertir la imagen a audio (1D)
        audio = imagen.flatten()

        # Calcular la STFT
        espectrograma = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

        # Obtener magnitud y fase
        magnitud = np.abs(espectrograma)
        fase = np.angle(espectrograma)

        # Guardar magnitud y fase en archivos .npy
        np.save(f'a-{nombre_base}_magnitud.npy', magnitud)
        np.save(f'a-{nombre_base}_fase.npy', fase)
        np.save(f'a-{nombre_base}_dimensiones.npy', (altura_original, ancho_original)) 
        print(f'Magnitud y fase guardados para a-{nombre_base}.')

        # Reconstruir el audio a partir del espectrograma
        espectrograma_reconstruido = magnitud * np.exp(1j * fase)
        audio_reconstruido = librosa.istft(espectrograma_reconstruido)

        # Normalizar el audio reconstruido
        audio_reconstruido /= np.max(np.abs(audio_reconstruido))

        # Guardar el audio como un archivo WAV
        sf.write(f'audio_de_{nombre_base}.wav', audio_reconstruido, 44100)
        print(f'Audio guardado como audio_de_{nombre_base}.wav.')

    except Exception as e:
        print(f"Error al procesar {ruta_imagen}: {e}")



ruta_imagen = '2.jpg'
nombre_base = 'imagen'
imagen_a_espectrograma(ruta_imagen, nombre_base)
