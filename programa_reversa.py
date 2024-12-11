
#bker face 2

reconstructed_image = np.zeros_like(new_image)

for i in range(len(n)):
    Ni = sum(n[:i])  
    ni = n[i] 
    
    for new_x in range(N):
        for new_y in range(N):
            x = int((ni / N) * (new_y - Ni) + Ni)
            y = int(((N / ni) * (new_x - Ni)) + (new_y % (N / ni)))

            x = x % N
            y = y % N

            reconstructed_image[x, y] = new_image[new_x, new_y]

cv2.imwrite("imagen_reconstruida.png", reconstructed_image)
