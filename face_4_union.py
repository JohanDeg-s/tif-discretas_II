import wave
import numpy as np

# Función para cargar un archivo WAV
def cargar_audio(ruta):
    with wave.open(ruta, 'rb') as archivo:
        params = archivo.getparams()
        frames = archivo.readframes(params.nframes)
        audio = np.frombuffer(frames, dtype=np.int16)
    return audio, params

# Función para guardar un archivo WAV
def guardar_audio(ruta, audio, params):
    with wave.open(ruta, 'wb') as archivo:
        archivo.setparams(params)
        archivo.writeframes(audio.tobytes())

# Rutas de los audios
ruta_espectrograma = 'audio2.wav'  # Audio generado a partir del espectrograma
ruta_generico = 'audio_generico.wav'  # Audio genérico

# Cargar audios
audio1, params1 = cargar_audio(ruta_espectrograma)
audio2, params2 = cargar_audio(ruta_generico)

# Validar que los parámetros coincidan
if params1.framerate != params2.framerate or params1.nchannels != params2.nchannels:
    raise ValueError("Los audios tienen diferentes frecuencias de muestreo o número de canales")

# Recortar o rellenar audios para que tengan la misma longitud
longitud = min(len(audio1), len(audio2))
audio1 = audio1[:longitud]
audio2 = audio2[:longitud]

# Combinar audios (suma y normalización)
audio_combinado = audio1 + audio2
audio_combinado = audio_combinado / np.max(np.abs(audio_combinado)) * 32767
audio_combinado = audio_combinado.astype(np.int16)

# Guardar el audio combinado
guardar_audio('audio_final.wav', audio_combinado, params1)
print("Audio combinado guardado como 'audio_final.wav'")