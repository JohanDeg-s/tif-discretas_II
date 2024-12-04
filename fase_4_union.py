import wave
import numpy as np

# Función para cargar un archivo WAV
def cargar_audio(ruta):
    with wave.open(ruta, 'rb') as archivo:
        params = archivo.getparams()
        frames = archivo.readframes(params.nframes)
        audio = np.frombuffer(frames, dtype=np.int16)
        if params.nchannels == 2:  # Si es estéreo
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)  # Convertir a mono
    return audio, params

# Función para guardar un archivo WAV
def guardar_audio(ruta, audio, params):
    with wave.open(ruta, 'wb') as archivo:
        archivo.setparams(params)
        archivo.writeframes(audio.tobytes())

# Función para cambiar la frecuencia de muestreo
def cambiar_frecuencia(audio, framerate_original, framerate_nuevo):
    factor = framerate_nuevo / framerate_original
    indices = np.arange(0, len(audio), 1 / factor)
    indices = indices[indices < len(audio)].astype(np.int32)  # Limitar los índices válidos
    audio_resampleado = audio[indices]
    return audio_resampleado

# Rutas de los audios
ruta_espectrograma = 'audio2.wav'  # Audio generado a partir del espectrograma
ruta_generico = 'va.wav'  # Audio genérico

# Cargar audios
audio1, params1 = cargar_audio(ruta_espectrograma)
audio2, params2 = cargar_audio(ruta_generico)

# Ajustar frecuencias de muestreo
framerate_comun = max(params1.framerate, params2.framerate)
audio1 = cambiar_frecuencia(audio1, params1.framerate, framerate_comun)
audio2 = cambiar_frecuencia(audio2, params2.framerate, framerate_comun)

# Recortar o rellenar audios para que tengan la misma longitud
longitud = min(len(audio1), len(audio2))
audio1 = audio1[:longitud]
audio2 = audio2[:longitud]

# Combinar audios (suma y normalización)
audio_combinado = audio1 + audio2
audio_combinado = audio_combinado / np.max(np.abs(audio_combinado)) * 32767
audio_combinado = audio_combinado.astype(np.int16)

# Guardar el audio combinado
params_comunes = params1._replace(framerate=framerate_comun, nchannels=1)
guardar_audio('audio_final.wav', audio_combinado, params_comunes)
print("Audio combinado guardado como 'audio_final.wav'")
