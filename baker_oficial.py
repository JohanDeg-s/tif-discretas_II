import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import hashlib


class CryptographicImageEncryption:
    def __init__(self, seed=None):
        """
        Advanced image encryption with improved cryptographic principles
        
        :param seed: Optional seed for reproducibility
        """
        # Use seed for consistent but unpredictable encryption
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
        
        # Set random seed
        np.random.seed(seed)
        self.seed = seed
    
    def _generate_cryptographic_key(self, image_shape):
        """
        Generate a cryptographic key based on image dimensions
        
        :param image_shape: Shape of the image
        :return: Cryptographic key matrix
        """
        # Use multiple sources of randomness
        np.random.seed(self.seed)
        
        # Create a more complex key generation
        key = np.random.randn(*image_shape)
        key = (key - key.min()) / (key.max() - key.min())
        
        return key
    
    def _advanced_diffusion(self, image, key):
        """
        Advanced pixel diffusion with multiple transformations
        
        :param image: Input image
        :param key: Cryptographic key
        :return: Diffused image
        """
        # Ensure consistent random state
        np.random.seed(self.seed)
        
        # Convert image to float for transformations
        diffused = image.copy().astype(np.float32) / 255.0
        
        # Complex non-linear transformations
        diffused = np.sin(diffused * key + key)
        diffused = np.log1p(np.abs(diffused))
        
        # Shuffle pixels 
        shuffle_indices = np.random.permutation(diffused.size)
        diffused = diffused.ravel()[shuffle_indices].reshape(diffused.shape)
        
        # Final normalization and scaling
        diffused = (diffused - diffused.min()) / (diffused.max() - diffused.min())
        
        return (diffused * 255).astype(np.uint8)
    
    def encrypt(self, image):
        """
        Encrypt image using advanced cryptographic principles
        
        :param image: Input image
        :return: Encrypted image
        """
        # Normalize input image
        normalized_image = image.astype(np.float32) / 255.0
        
        # Generate cryptographic key matching image shape
        key = self._generate_cryptographic_key(normalized_image.shape)
        
        # Apply advanced diffusion
        encrypted = self._advanced_diffusion(image, key)
        
        return encrypted
    
    def decrypt(self, encrypted_image):
        """
        Decrypt image with improved reconstruction
        
        :param encrypted_image: Encrypted image
        :return: Reconstructed original image
        """
        # Regenerate key
        key = self._generate_cryptographic_key(encrypted_image.shape)
        
        # Reverse transformations
        np.random.seed(self.seed)
        
        # Reverse pixel shuffle
        reverse_shuffle = np.argsort(np.random.permutation(encrypted_image.size))
        
        # Convert to float and normalize
        decrypted = encrypted_image.astype(np.float32) / 255.0
        
        # Reverse shuffle
        decrypted = decrypted.ravel()[reverse_shuffle].reshape(decrypted.shape)
        
        # Approximate reverse transformations
        decrypted = np.exp(decrypted) - 1
        decrypted = np.arcsin(decrypted / (1 + key))
        
        # Normalize and scale back to original range
        decrypted = (decrypted - decrypted.min()) / (decrypted.max() - decrypted.min())
        
        return (decrypted * 255).astype(np.uint8)

def plot_comprehensive_analysis(original, encrypted):
    """
    Comprehensive visualization and analysis
    
    :param original: Original image
    :param encrypted: Encrypted image
    """
    plt.figure(figsize=(16, 10))
    
    # Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    
    # Encrypted Image
    plt.subplot(2, 3, 2)
    plt.imshow(encrypted, cmap='gray')
    plt.title('Encrypted Image')
    
    # Original Histogram
    plt.subplot(2, 3, 3)
    plt.hist(original.ravel(), bins=256, range=[0, 256], density=True, alpha=0.7, color='blue')
    plt.title('Original Image Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    
    # Encrypted Histogram
    plt.subplot(2, 3, 4)
    plt.hist(encrypted.ravel(), bins=256, range=[0, 256], density=True, alpha=0.7, color='red')
    plt.title('Encrypted Image Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    
    # Histogram Comparison
    original_hist, _ = np.histogram(original, bins=256, range=[0, 256], density=True)
    encrypted_hist, _ = np.histogram(encrypted, bins=256, range=[0, 256], density=True)
    
    plt.subplot(2, 3, 5)
    plt.plot(original_hist, label='Original', color='blue')
    plt.plot(encrypted_hist, label='Encriptado', color='red')
    plt.title('Comparación de Histogramas')
    plt.xlabel('Intensidad de Pixel')
    plt.ylabel('Frecuencia normalizada')
    plt.legend()
    
    # Entropy Calculation
    def calculate_entropy(image):
        """Calculate image entropy"""
        hist, _ = np.histogram(image, bins=256, range=[0, 256], density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    original_entropy = calculate_entropy(original)
    encrypted_entropy = calculate_entropy(encrypted)
    
    plt.subplot(2, 3, 6)
    plt.text(0.5, 0.5, 
             f'Entropy Analysis:\n\n'
             f'Original Image: {original_entropy:.2f}\n'
             f'Encrypted Image: {encrypted_entropy:.2f}', 
             horizontalalignment='center', 
             verticalalignment='center',
             fontsize=10)
    plt.title('Entropy Comparison')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Ensure directories exist
    os.makedirs('test_images', exist_ok=True)
    os.makedirs('RESULTADOS', exist_ok=True)
    os.makedirs('Decrypted', exist_ok=True)  # New directory for decrypted images
    
    # Create a random test image if no images exist
    if not os.listdir('test_images'):
        test_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        plt.imsave('test_images/test_image.png', test_image, cmap='gray')
    
    # Test images list
    test_images = [
        os.path.join('test_images', img) for img in os.listdir('test_images') 
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]
    
    # Encryption
    encryption = CryptographicImageEncryption()
    
    for image_path in test_images:
        try:
            # Read image in grayscale
            original_image = np.array(Image.open(image_path).convert('L'))
            
            # Encrypt
            encrypted_image = encryption.encrypt(original_image)
            
            # Comprehensive analysis
            plot_comprehensive_analysis(original_image, encrypted_image)
            
            # Save encrypted image
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            encrypted_filename = os.path.join('RESULTADOS', f'{base_filename}_cryptographic_encrypted.png')
            plt.imsave(encrypted_filename, encrypted_image, cmap='gray')
            
            # Decrypt and save
            decrypted_image = encryption.decrypt(encrypted_image)
            decrypted_filename = os.path.join('Decrypted', f'{base_filename}_decrypted.png')
            plt.imsave(decrypted_filename, decrypted_image, cmap='gray')
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# Desencriptar la imagen######################################################################
#encryption = CryptographicImageEncryption(seed=42)  # Usa la misma semilla que se usó para encriptar

# Cargar la imagen encriptada
#encrypted_image = np.array(Image.open('ruta_a_tu_imagen_encriptada.png').convert('L'))

# Desencriptar la imagen
#decrypted_image = encryption.decrypt(encrypted_image)

# Guardar la imagen desencriptada
#plt.imsave('imagen_desencriptada.png', decrypted_image, cmap='gray')
if __name__ == "__main__":
    main()
