import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft

# Función para incrustar datos RGB en un archivo de audio usando FFT
def embed_rgb_in_audio_fft(audio_filename, output_filename, rgb_data, width, height):
    # Leer archivo de audio
    sample_rate, audio_data = wavfile.read(audio_filename)
    
    # Convertir audio estéreo a mono si es necesario
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    # Aplicar FFT al audio
    audio_fft = fft(audio_data)

    # Incrustar datos RGB en las frecuencias altas del audio
    idx = 0  # Índice para los datos RGB
    for y in range(height):
        for x in range(width):
            if idx < len(rgb_data):
                r, g, b = rgb_data[idx]
                index = len(audio_fft) - (idx + 1) * 3
                if index >= 2:  # Asegurarse de que hay suficientes índices
                    audio_fft[index] += r * 1e-1  # Ajuste de escalado
                    audio_fft[index - 1] += g * 1e-1
                    audio_fft[index - 2] += b * 1e-1
                idx += 1

    # Reconstruir el audio usando IFFT
    modified_audio = ifft(audio_fft).real

    # Normalizar el audio a rango de 16 bits
    modified_audio = np.int16(modified_audio / np.max(np.abs(modified_audio)) * 32767)
    
    # Guardar el archivo de audio modificado
    wavfile.write(output_filename, sample_rate, modified_audio)

# Función para extraer datos RGB de un archivo de audio usando FFT
def extract_rgb_from_audio_fft(audio_filename, output_rgb_file, width, height):
    # Leer archivo de audio
    sample_rate, audio_data = wavfile.read(audio_filename)
    
    # Convertir audio estérepipo a mono si es necesario
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    # Aplicar FFT al audio
    audio_fft = fft(audio_data)

    # Inicializar la lista de datos RGB
    rgb_data = []

    # Extraer datos RGB de las frecuencias altas
    num_colors = min(len(audio_fft) // 3, width * height)  # Limitar a tamaño de imagen
    idx = 0  # Índice para los datos RGB
    for y in range(height):
        for x in range(width):
            if idx < num_colors:
                index = len(audio_fft) - (idx + 1) * 3
                if index >= 2:  # Asegurarse de que hay suficientes valores
                    r = int(np.clip(audio_fft[index].real * 1e3, 0, 255))  # Escalar y limitar
                    g = int(np.clip(audio_fft[index - 1].real * 1e3, 0, 255))
                    b = int(np.clip(audio_fft[index - 2].real * 1e3, 0, 255))
                    rgb_data.append((r, g, b))
                idx += 1

    # Guardar los valores RGB en un archivo de texto
    with open(output_rgb_file, 'w') as file:
        for i, (r, g, b) in enumerate(rgb_data):
            file.write(f"Pixel ({i % width}, {i // width}) - R: {r}, G: {g}, B: {b}\n")

# Ejemplo de uso:
# Datos RGB a incrustar (imagen de 16x16 píxeles)
# help ya no entiendo bien los codigos en python
rgb_data = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] * 16 * 16  # Imagen ficticia

# Incrustar los datos RGB en el archivo de audio
embed_rgb_in_audio_fft("audio.wav", "audio_modificado.wav", rgb_data, 16, 16)

# Extraer los datos RGB del archivo de audio modificado
extract_rgb_from_audio_fft("audio_modificado.wav", "rgb_extraidos.txt", 16, 16)
