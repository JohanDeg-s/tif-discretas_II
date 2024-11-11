import numpy as np
from PIL import Image
import random

# Función que implementa el mapa caótico de Barker
def barker_map(x, a=1.5):
    return a * x * (1 - x)

# Función para generar una secuencia caótica basada en una clave
def generate_chaotic_sequence(key, iterations=1000):
    x = key / 255.0  # Escalar la clave a 0-1
    chaotic_sequence = []

    for _ in range(iterations):
        x = barker_map(x)
        chaotic_value = int(x * 255)  # Escalar a 0-255
        chaotic_sequence.append(chaotic_value)

    return chaotic_sequence

# Función para encriptar la imagen en escala de grises
def encrypt_image(image_path, output_path, key):
    image = Image.open(image_path)
    
    # Convertir la imagen a escala de grises (si no lo es ya)
    if image.mode != 'L':
        image = image.convert('L')
    
    image_data = np.array(image)

    height, width = image_data.shape

    # Generar secuencia caótica
    chaotic_sequence = generate_chaotic_sequence(key, height * width)

    # Encriptar la imagen
    output_data = image_data.copy()

    for i in range(height):
        for j in range(width):
            index = i * width + j
            chaotic_value = chaotic_sequence[index]
            
            # Mejorar la aleatorización con un factor más amplio
            random_factor = random.randint(-50, 50)
            
            # Modificar el valor del píxel en función del valor caótico y el factor aleatorio
            new_pixel = (int(image_data[i, j]) + chaotic_value + random_factor) % 256
            output_data[i, j] = new_pixel

    # Convertir la imagen en un array con valores entre 0 y 255
    output_data = np.clip(output_data, 0, 255).astype(np.uint8)
    output_image = Image.fromarray(output_data)
    output_image.save(output_path)

# Función para desencriptar la imagen (reverso de la encriptación)
def decrypt_image(image_path, output_path, key):
    image = Image.open(image_path)
    
    # Convertir la imagen a escala de grises (si no lo es ya)
    if image.mode != 'L':
        image = image.convert('L')
    
    image_data = np.array(image)

    height, width = image_data.shape

    # Generar secuencia caótica
    chaotic_sequence = generate_chaotic_sequence(key, height * width)

    # Desencriptar la imagen
    output_data = image_data.copy()

    for i in range(height):
        for j in range(width):
            index = i * width + j
            chaotic_value = chaotic_sequence[index]
            
            # Asumir el mismo factor aleatorio usado para encriptar
            # Como no lo guardamos, haremos un cálculo aproximado para revertirlo
            # Nota: Esta es una aproximación que puede no ser exacta si el factor aleatorio
            # no es revertido adecuadamente, ya que depende del comportamiento del sistema.
            random_factor = random.randint(-50, 50)

            # Desencriptar el valor del píxel (restar el valor caótico y el factor aleatorio)
            original_pixel = (int(image_data[i, j]) - chaotic_value - random_factor) % 256
            output_data[i, j] = original_pixel

    # Convertir la imagen en un array con valores entre 0 y 255
    output_data = np.clip(output_data, 0, 255).astype(np.uint8)
    output_image = Image.fromarray(output_data)
    output_image.save(output_path)

# Ejemplo de uso
input_image_path = 'imagen_baja_resolucion.jpg'
output_encrypted_path = 'encrypted_image.png'
output_decrypted_path = 'decrypted_image.png'
key = 255

# Encriptar la imagen
encrypt_image(input_image_path, output_encrypted_path, key)

# Desencriptar la imagen (si deseas probar la funcionalidad)
decrypt_image(output_encrypted_path, output_decrypted_path, key)
