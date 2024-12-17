import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import os

class ChaoticImageEncryption:
    def __init__(self, r=3.93, x0=0.732, a=0.321, b=0.823):

        self.r = r 
        self.x0 = x0  
        
     
        self.a = a
        self.b = b
    
    def logistic_map_sequence(self, size):
       
        rows, cols = size
        sequence = np.zeros((rows, cols), dtype=np.float32)
        x = self.x0
        
        for i in range(rows):
            for j in range(cols):
               
                x = self.r * x * (1 - x)
                
        
                sequence[i, j] = 1 if x > 0.5 else 0
        
        return sequence
    
    def baker_map_sequence(self, size):
       
        rows, cols = size
        sequence = np.zeros((rows, cols), dtype=np.float32)
        
        for i in range(rows):
            for j in range(cols):
               
                if self.a < 0.5:
                    x_new = 2 * self.a
                    y_new = self.b / 2
                else:
                    x_new = 2 - 2 * self.a
                    y_new = 1 - self.b / 2
                
                
                sequence[i, j] = 1 if y_new > 0.5 else 0
        
        return sequence
    
    def encrypt(self, image):
 
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

    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    max_pixel = 255.0
    return 10 * np.log10((max_pixel ** 2) / mse)

def main():
    
        # Cargar imagen encriptada
    encrypted_image_path = r"C:\Users\USER\Documents\ultimo tif\RESULTADOS\cameraman_encrypted.png"
    
    # Leer la imagen encriptada
    encrypted_image = np.array(Image.open(encrypted_image_path).convert('L'))
    
    # Configurar parámetros de encriptación (si es necesario)
    encryption = ChaoticImageEncryption(
        r=3.93,      # Logistic map parameter
        x0=0.732,    # Initial condition
        a=0.321,     # Baker map parameter
        b=0.823      # Baker map parameter
    )
    
    try:
        # Desencriptar la imagen
        decrypted_image = encryption.decrypt(encrypted_image)
        
        # Guardar imagen desencriptada
        decrypted_filename = "imagen_desencriptada.png"
        plt.imsave(decrypted_filename, decrypted_image, cmap='gray')
        
        # Mostrar la imagen desencriptada
        plt.imshow(decrypted_image, cmap='gray')
        plt.title('Desencriptada')
        plt.show()
        
    except Exception as e:
        print(f"Error procesando {encrypted_image_path}: {e}")
#fin DESECNRIPTADO

    os.makedirs('test_images', exist_ok=True)
    os.makedirs('RESULTADOS', exist_ok=True)
    

    if not os.listdir('test_images'):
        test_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        plt.imsave('test_images/test_image.png', test_image, cmap='gray')
    

    test_images = [
        os.path.join('test_images', img) for img in os.listdir('test_images') 
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]
    
    encryption = ChaoticImageEncryption(
        r=3.93,      # Logistic map parameter
        x0=0.732,    # Initial condition
        a=0.321,     # Baker map parameter
        b=0.823      # Baker map parameter
    )

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
