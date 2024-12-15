import numpy as np
from PIL import Image
import librosa
import matplotlib.pyplot as plt

def audio_a_espectrograma(ruta_audio, n_fft=2048, hop_length=512):
    try:
        # Cargar el archivo de audio
        y, sr = librosa.load(ruta_audio, sr=None)  # Mantener la frecuencia de muestreo original
        
        # Normalizar la señal de audio
        y /= np.max(np.abs(y))

        # Calcular la STFT
        espectrograma = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        
        # Obtener magnitud y fase
        magnitud = np.abs(espectrograma)
        fase = np.angle(espectrograma)
        
        return magnitud, fase
    except Exception as e:
        print(f"Error al procesar {ruta_audio}: {e}")
        return None, None

def visualizar_espectrogramas(magnitud_imagen, magnitud_audio):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Espectrograma de la Imagen')
    plt.imshow(np.log(1 + magnitud_imagen), aspect='auto', cmap='gray')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title('Espectrograma del Audio')
    plt.imshow(np.log(1 + magnitud_audio), aspect='auto', cmap='gray')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def reconstruir_imagen(magnitud, fase, dimensiones_originales):
    espectrograma_reconstruido = magnitud * np.exp(1j * fase)
    audio_reconstruido = librosa.istft(espectrograma_reconstruido)

    # Reescalar el audio reconstruido a las dimensiones originales
    imagen_reconstruida = np.resize(audio_reconstruido, dimensiones_originales)
    return imagen_reconstruida

def normalizar_imagen(imagen):
    imagen -= imagen.min()
    imagen /= imagen.max()
    imagen *= 255
    return imagen.astype(np.uint8)

# Cargar el espectrograma de la imagen desde el archivo .npy
magnitud_imagen = np.load('a-imagen_magnitud.npy')
fase_imagen = np.load('a-imagen_fase.npy')
dimensiones_originales = np.load('a-imagen_dimensiones.npy')  # Cargar dimensiones originales

# Reconstruir la imagen
imagen_reconstruida = reconstruir_imagen(magnitud_imagen, fase_imagen, dimensiones_originales)

# Normalizar y guardar la imagen reconstruida
imagen_reconstruida_normalizada = normalizar_imagen(imagen_reconstruida)
Image.fromarray(imagen_reconstruida_normalizada).save('a-imagen_reconstruida.png')
print('Imagen reconstruida guardada como imagen_reconstruida.png.')





