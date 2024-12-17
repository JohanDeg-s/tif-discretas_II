import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import os

class ChaoticImageEncryption:
    def __init__(self, r=3.93, x0=0.732, a=0.321, b=0.823):
        """
        Initialize encryption parameters based on the research paper
        
        :param r: Logistic map control parameter (default from paper)
        :param x0: Logistic map initial condition (default from paper)
        :param a: Baker map parameter a
        :param b: Baker map parameter b
        """
        # Logistic Map Parameters
        self.r = r  # Control parameter (between 3.56 and 3.82 for chaotic behavior)
        self.x0 = x0  # Initial condition
        
        # Baker Map Parameters
        self.a = a
        self.b = b
    
    def logistic_map_sequence(self, size):
        """
        Generate pseudo-random sequence using Logistic Map
        
        :param size: Size of the sequence (tuple of rows, columns)
        :return: Numpy array of binary sequence
        """
        rows, cols = size
        sequence = np.zeros((rows, cols), dtype=np.float32)
        x = self.x0
        
        for i in range(rows):
            for j in range(cols):
                # Logistic map equation: X(n+1) = r * X(n) * (1 - X(n))
                x = self.r * x * (1 - x)
                
                # Quantization as per the paper's equation
                sequence[i, j] = 1 if x > 0.5 else 0
        
        return sequence
    
    def baker_map_sequence(self, size):
        """
        Generate pseudo-random sequence using Baker Map
        
        :param size: Size of the sequence (tuple of rows, columns)
        :return: Numpy array of binary sequence
        """
        rows, cols = size
        sequence = np.zeros((rows, cols), dtype=np.float32)
        
        for i in range(rows):
            for j in range(cols):
                # Baker map transformation as described in the paper
                if self.a < 0.5:
                    x_new = 2 * self.a
                    y_new = self.b / 2
                else:
                    x_new = 2 - 2 * self.a
                    y_new = 1 - self.b / 2
                
                # Quantization
                sequence[i, j] = 1 if y_new > 0.5 else 0
        
        return sequence
    
    def encrypt(self, image):
        """
        Encrypt the image using two-stage chaotic encryption
        
        :param image: Input image as numpy array
        :return: Encrypted image
        """
        # Step 1: First encryption using Logistic Map
        logistic_sequence = self.logistic_map_sequence(image.shape)
        logistic_encrypted = np.bitwise_xor(
            image.astype(np.uint8), 
            (logistic_sequence * 255).astype(np.uint8)
        )
        
        # Step 2: Second encryption using Baker Map
        baker_sequence = self.baker_map_sequence(image.shape)
        final_encrypted = np.bitwise_xor(
            logistic_encrypted, 
            (baker_sequence * 255).astype(np.uint8)
        )
        
        return final_encrypted
    
    def decrypt(self, encrypted_image):
        """
        Decrypt the image using reverse of encryption process
        
        :param encrypted_image: Encrypted image as numpy array
        :return: Decrypted image
        """
        # Reverse Baker Map encryption
        baker_sequence = self.baker_map_sequence(encrypted_image.shape)
        baker_decrypted = np.bitwise_xor(
            encrypted_image.astype(np.uint8), 
            (baker_sequence * 255).astype(np.uint8)
        )
        
        # Reverse Logistic Map encryption
        logistic_sequence = self.logistic_map_sequence(encrypted_image.shape)
        decrypted_image = np.bitwise_xor(
            baker_decrypted, 
            (logistic_sequence * 255).astype(np.uint8)
        )
        
        return decrypted_image

def histogram_analysis(original_image, encrypted_image):
    """
    Perform histogram analysis as described in the paper
    
    :param original_image: Original image numpy array
    :param encrypted_image: Encrypted image numpy array
    """
    plt.figure(figsize=(12, 6))
    
    # Original Image Histogram
    plt.subplot(2, 2, 1)
    plt.hist(original_image.ravel(), 256, [0, 256])
    plt.title('Original Image Histogram')
    
    # Encrypted Image Histogram
    plt.subplot(2, 2, 2)
    plt.hist(encrypted_image.ravel(), 256, [0, 256])
    plt.title('Encrypted Image Histogram')
    
    # Original Image
    plt.subplot(2, 2, 3)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    
    # Encrypted Image
    plt.subplot(2, 2, 4)
    plt.imshow(encrypted_image, cmap='gray')
    plt.title('Encrypted Image')
    
    plt.tight_layout()
    plt.show()

def correlation_analysis(image):
    """
    Compute correlation between adjacent pixels
    
    :param image: Input image numpy array
    :return: Correlation coefficients
    """
    def compute_correlation(x, y):
        """Compute correlation coefficient between two arrays"""
        return np.corrcoef(x.ravel(), y.ravel())[0, 1]
    
    correlations = {}
    
    # Horizontal correlation
    x_horizontal = image[:, :-1]
    y_horizontal = image[:, 1:]
    correlations['Horizontal'] = compute_correlation(x_horizontal, y_horizontal)
    
    # Vertical correlation
    x_vertical = image[:-1, :]
    y_vertical = image[1:, :]
    correlations['Vertical'] = compute_correlation(x_vertical, y_vertical)
    
    # Diagonal correlation
    x_diagonal = image[:-1, :-1]
    y_diagonal = image[1:, 1:]
    correlations['Diagonal'] = compute_correlation(x_diagonal, y_diagonal)
    
    return correlations

def calculate_psnr(original, reconstructed):
    """
    Calculate Peak Signal to Noise Ratio (PSNR)
    
    :param original: Original image numpy array
    :param reconstructed: Reconstructed image numpy array
    :return: PSNR value
    """
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    max_pixel = 255.0
    return 10 * np.log10((max_pixel ** 2) / mse)

def main():
    # Ensure test images and results directories exist
    os.makedirs('test_images', exist_ok=True)
    os.makedirs('RESULTADOS', exist_ok=True)
    
    # Create a random test image if no images exist
    if not os.listdir('test_images'):
        test_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        plt.imsave('test_images/test_image.png', test_image, cmap='gray')
    
    # Test images list
    test_images = [
        os.path.join('test_images', img) for img in os.listdir('test_images') 
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]
    
    # Encryption parameters from the paper
    encryption = ChaoticImageEncryption(
        r=3.93,      # Logistic map parameter
        x0=0.732,    # Initial condition
        a=0.321,     # Baker map parameter
        b=0.823      # Baker map parameter
    )
    
    # Process each image
    for image_path in test_images:
        try:
            # Read image in grayscale
            original_image = np.array(Image.open(image_path).convert('L'))
            
            # Encrypt
            encrypted_image = encryption.encrypt(original_image)
            
            # Decrypt
            decrypted_image = encryption.decrypt(encrypted_image)
            
            # Save encrypted image to RESULTADOS folder
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            encrypted_filename = os.path.join('RESULTADOS', f'{base_filename}_encrypted.png')
            plt.imsave(encrypted_filename, encrypted_image, cmap='gray')
            
            # Visualization
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(original_image, cmap='gray')
            plt.title('Original Image')
            
            plt.subplot(1, 3, 2)
            plt.imshow(encrypted_image, cmap='gray')
            plt.title('Encrypted Image')
            
            plt.subplot(1, 3, 3)
            plt.imshow(decrypted_image, cmap='gray')
            plt.title('Decrypted Image')
            
            plt.tight_layout()
            plt.show()
            
            # Analyses
            print(f"\nAnalysis for {os.path.basename(image_path)}:")
            
            # Histogram Analysis
            histogram_analysis(original_image, encrypted_image)
            
            # Correlation Analysis
            corr_results = correlation_analysis(original_image)
            print("Correlation Coefficients:")
            for direction, corr in corr_results.items():
                print(f"{direction}: {corr}")
            
            # PSNR Calculation
            psnr = calculate_psnr(original_image, decrypted_image)
            print(f"PSNR: {psnr} dB")
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    main()
