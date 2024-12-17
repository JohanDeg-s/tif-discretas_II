import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
from scipy.interpolate import interp1d

# 1. Leer los audios
def read_audio_files(song_path, spectrogram_audio_path):
    # Leer la canción genérica
    sample_rate_song, song_audio = wavfile.read(song_path)
    # Leer el audio generado del espectrograma
    sample_rate_spec, spec_audio = wavfile.read(spectrogram_audio_path)
    
    # Asegurar que ambos audios tengan la misma tasa de muestreo
    if sample_rate_song != sample_rate_spec:
        raise ValueError("Las tasas de muestreo de los audios no coinciden.")
    
    # Convertir a mono si son estéreo
    if len(song_audio.shape) > 1:
        song_audio = np.mean(song_audio, axis=1).astype(song_audio.dtype)
    if len(spec_audio.shape) > 1:
        spec_audio = np.mean(spec_audio, axis=1).astype(spec_audio.dtype)
    
    return sample_rate_song, song_audio, spec_audio

# 2. Ajustar tamaños de audio
def match_audio_lengths(song_audio, spec_audio):
    min_length = min(len(song_audio), len(spec_audio))
    return song_audio[:min_length], spec_audio[:min_length]

# 3. Ocultar el espectrograma en la canción
def embed_audio_in_frequencies(song_audio, spec_audio):
    # Aplicar FFT a la canción
    song_fft = fft(song_audio)
    # Aplicar FFT al audio del espectrograma
    spec_fft = fft(spec_audio)
    
    # Usar las frecuencias altas del espectro de la canción para ocultar el espectrograma
    start_idx = int(0.75 * len(song_fft))  # Último 25% de las frecuencias
    
    # Calcular el espacio disponible en las frecuencias altas
    available_space = len(song_fft) - start_idx
    
    if len(spec_fft) > available_space:
        # Escalar el espectrograma para que encaje en el espacio disponible
        interpolation_function = interp1d(
            np.linspace(0, 1, len(spec_fft)),
            spec_fft,
            kind="linear",
            axis=0
        )
        spec_fft_scaled = interpolation_function(np.linspace(0, 1, available_space))
    else:
        # Si cabe, usar el espectrograma directamente
        spec_fft_scaled = spec_fft

    # Incrustar el espectrograma escalado en las frecuencias altas
    song_fft[start_idx:start_idx + len(spec_fft_scaled)] = (
        song_fft[start_idx:start_idx + len(spec_fft_scaled)].real + spec_fft_scaled.real
        + 1j * song_fft[start_idx:start_idx + len(spec_fft_scaled)].imag
    )
    
    # Reconstruir la canción modificada
    song_modified = np.real(ifft(song_fft))
    return song_modified

# 4. Guardar el audio modificado
def save_audio(output_path, sample_rate, audio_modified):
    audio_modified = np.clip(audio_modified, -32768, 32767).astype(np.int16)  # Asegurar formato de 16 bits
    wavfile.write(output_path, sample_rate, audio_modified)

# Main
if __name__ == "__main__":
    song_path = "musica.wav"           # Ruta del audio de la canción genérica
    spectrogram_audio_path = "0.1.wav"  # Ruta del audio del espectrograma
    output_path = "cancion_modificada.wav"       # Ruta para guardar el audio modificado
    
    # Leer los audios
    sample_rate, song_audio, spec_audio = read_audio_files(song_path, spectrogram_audio_path)
    # Ajustar los tamaños de los audios
    song_audio, spec_audio = match_audio_lengths(song_audio, spec_audio)
    # Ocultar el espectrograma en las frecuencias altas de la canción
    song_modified = embed_audio_in_frequencies(song_audio, spec_audio)
    # Guardar el audio modificado
    save_audio(output_path, sample_rate, song_modified)
    print("Audio modificado guardado en:", output_path)
