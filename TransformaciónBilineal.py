import cv2
import numpy as np

# Cargar la imagen de entrada
input_image = cv2.imread('chop.jpg')

# Verificar que la imagen se haya cargado correctamente
if input_image is None:
    print("Error al cargar la imagen")
    exit()

# Especificar el tamaño de la imagen transformada
new_width = input_image.shape[1]
new_height = input_image.shape[0]

# Calcular la matriz de transformación bilineal
src_points = np.float32([[0, 0], [new_width - 1, 0], [0, new_height - 1], [new_width - 1, new_height - 1]])

# Definir los puntos de destino para formar un trapecio
dst_points = np.float32([[new_width * 0.2, 0], [new_width * 0.8, 0], [0, new_height - 1], [new_width - 1, new_height - 1]])

# Calcular la matriz de transformación
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Aplicar la transformación bilineal
output_image = cv2.warpPerspective(input_image, matrix, (new_width, new_height), flags=cv2.INTER_LINEAR)

# Guardar la imagen transformada
cv2.imwrite('output.jpg', output_image)

# Mostrar la imagen original y la transformada
cv2.imshow('Input Image', input_image)
cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
