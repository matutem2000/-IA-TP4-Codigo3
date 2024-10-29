import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Subir el archivo 
uploaded = files.upload()

# Usar el nombre del archivo subido para cargar la imagen
image_path = next(iter(uploaded))  # Obtiene el nombre del primer archivo subido
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Aplicar desenfoque para reducir el ruido
blurred_image = cv2.GaussianBlur(image, (9, 9), 2)

# Usar la función de OpenCV para la Transformada de Hough de circunferencias
# Aquí se detectan circunferencias de un radio entre 20 y 50 píxeles
circles = cv2.HoughCircles(blurred_image, 
                           cv2.HOUGH_GRADIENT, 
                           dp=1.2, 
                           minDist=50,
                           param1=50,
                           param2=30, 
                           minRadius=20, 
                           maxRadius=50)

# Crear una copia de la imagen original para dibujar los círculos detectados
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Dibujar los círculos detectados en la imagen
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        # Dibujar el círculo en la imagen
        cv2.circle(output_image, (x, y), r, (0, 255, 0), 2)
        # Dibujar el centro del círculo
        cv2.circle(output_image, (x, y), 2, (0, 0, 255), 3)

# Mostrar la imagen con las circunferencias detectadas
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("Circunferencias detectadas")
plt.axis("off")
plt.show()

