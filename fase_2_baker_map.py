import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
#crear una carpeta test_images, y guardar todas las imagenes ahi
#las encriptaciones seran guardadas en una nueva carpeta llamada RESULTADOS
#para desecnriptar una imagen en especifico, usar las ultimas lineas de codigo 
class ChaoticImageEncryption:
    def __init__(self, r=3.93, x0=0.732, a=0.321, b=0.823):

        self.r = r 
        self.x0 = x0  
        self.a = a
        self.b = b
    
    def logistic_map_sequence(self, size):
 
        filas, cols = size
        sequence = np.zeros((filas, cols), dtype=np.float32)
        x = self.x0
        
        for i in range(filas):
            for j in range(cols):
                x = self.r * x * (1 - x)
                sequence[i, j] = 1 if x > 0.5 else 0
        
        return sequence
    
    def baker_map_sequence(self, size):

        filas, cols = size
        sequence = np.zeros((filas, cols), dtype=np.float32)
        
        for i in range(filas):
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

        logistic_sequence = self.logistic_map_sequence(image.shape)
        logistic_encriptado = np.bitwise_xor(
            image.astype(np.uint8), 
            (logistic_sequence * 255).astype(np.uint8)
        )
        
       
        baker_sequence = self.baker_map_sequence(image.shape)
        final_encriptado = np.bitwise_xor(
            logistic_encriptado, 
            (baker_sequence * 255).astype(np.uint8)
        )
        
        return final_encriptado
    
    def decrypt(self, encriptado_image):

        baker_sequence = self.baker_map_sequence(encriptado_image.shape)
        baker_desencriptado = np.bitwise_xor(
            encriptado_image.astype(np.uint8), 
            (baker_sequence * 255).astype(np.uint8)
        )

        logistic_sequence = self.logistic_map_sequence(encriptado_image.shape)
        img_desencriptada = np.bitwise_xor(
            baker_desencriptado, 
            (logistic_sequence * 255).astype(np.uint8)
        )
        
        return img_desencriptada

def Histograma_analysis(original_image, encriptado_image):

    plt.figure(figsize=(12, 6))
    

    plt.subplot(2, 2, 1)
    plt.hist(original_image.ravel(), 256, [0, 256])
    plt.title('Original Image Histograma')
    

    plt.subplot(2, 2, 2)
    plt.hist(encriptado_image.ravel(), 256, [0, 256])
    plt.title('encriptado Image Histograma')
    

    plt.subplot(2, 2, 3)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    

    plt.subplot(2, 2, 4)
    plt.imshow(encriptado_image, cmap='gray')
    plt.title('encriptado Image')
    
    plt.tight_layout()
    plt.show()

def correlation_analysis(image):

    def compute_correlation(x, y):
        
        return np.corrcoef(x.ravel(), y.ravel())[0, 1]
    
    correlations = {}
    

    x_horizontal = image[:, :-1]
    y_horizontal = image[:, 1:]
    correlations['Horizontal'] = compute_correlation(x_horizontal, y_horizontal)
    

    x_vertical = image[:-1, :]
    y_vertical = image[1:, :]
    correlations['Vertical'] = compute_correlation(x_vertical, y_vertical)
    

    x_diagonal = image[:-1, :-1]
    y_diagonal = image[1:, 1:]
    correlations['Diagonal'] = compute_correlation(x_diagonal, y_diagonal)
    
    return correlations

def calculate_psnr(original, reconstructed):

    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    max_pixel = 255.0
    return 10 * np.log10((max_pixel ** 2) / mse)

def main():
    os.makedirs('test_images', exist_ok=True)
    os.makedirs('RESULTADOS', exist_ok=True)

    if not os.listdir('test_images'):
        test_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        plt.imsave('test_images/test_image.png', test_image, cmap='gray')

    test_images = [
        os.path.join('test_images', img) for img in os.listdir('test_images') 
        if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]

    encryption_params = {
        'r': 3.99,     
        'x0': 0.732,    
        'a': 0.321,     
        'b': 0.823      
    }
    
    for image_path in test_images:
        try:

            original_image = np.array(Image.open(image_path).convert('L'))
            
 
            encryption = ChaoticImageEncryption(**encryption_params)

            encriptado_image = encryption.encrypt(original_image)
            img_desencriptada = encryption.decrypt(encriptado_image)

            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            encriptado_filename = os.path.join('RESULTADOS', f'{base_filename}_encriptado.png')
            plt.imsave(encriptado_filename, encriptado_image, cmap='gray')
            
  
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(original_image, cmap='gray')
            plt.title('Original Image')
            
            plt.subplot(1, 3, 2)
            plt.imshow(encriptado_image, cmap='gray')
            plt.title('encriptado Image')
            
            plt.subplot(1, 3, 3)
            plt.imshow(img_desencriptada, cmap='gray')
            plt.title('desencriptado Image')
            
            plt.tight_layout()
            plt.show()
            
 
            print(f"\nanalisis para {os.path.basename(image_path)}:")

            Histograma_analysis(original_image, encriptado_image)

            corr_results = correlation_analysis(original_image)
            print("Correlation Coefficients:")
            for direction, corr in corr_results.items():
                print(f"{direction}: {corr}")
            

            psnr = calculate_psnr(original_image, img_desencriptada)
            print(f"PSNR: {psnr} dB")
        
        except Exception as e:
            print(f"Error{image_path}: {e}")
#des


if __name__ == "__main__":
    main()
#SI SE NECESITA DESECNRIPTAR UNA IMAGEN 
"""
imagen_encriptada = np.array(Image.open("RESULTADOS\\mandrill_encrypted.png").convert('L'))
encryption = ChaoticImageEncryption(r=3.99, x0=0.732, a=0.321, b=0.823)
imagen_desencriptada = encryption.decrypt(imagen_encriptada)
plt.imsave("desencriptado.jpg", imagen_desencriptada, cmap='gray')
"""
