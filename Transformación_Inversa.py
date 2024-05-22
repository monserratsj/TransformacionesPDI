import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
image = cv2.imread('2.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Dimensiones de la imagen
h, w = image.shape[:2]

# Puntos de origen (cuatro esquinas de la imagen)
pts_src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                   dtype='float32')

# Puntos de destino para la transformación de perspectiva
pts_dst = np.array(
    [[0, 0], [w - 1, 0], [int(0.6 * w), h - 1], [int(0.4 * w), h - 1]],
    dtype='float32')

# Calcular la matriz de transformación de perspectiva
M = cv2.getPerspectiveTransform(pts_src, pts_dst)

# Aplicar la transformación de perspectiva
image_perspective = cv2.warpPerspective(image_rgb, M, (w, h))


# Introducir interpolación
def add_interpolation_artifacts(image, scale_factor=2):
    height, width = image.shape[:2]
    new_width = width // scale_factor
    new_height = height // scale_factor

    small = cv2.resize(image, (new_width, new_height),
                       interpolation=cv2.INTER_LINEAR)
    enlarged = cv2.resize(small, (width, height),
                          interpolation=cv2.INTER_NEAREST)
    return enlarged


# Aplicar la transformación inversa
M_inv = cv2.getPerspectiveTransform(pts_dst, pts_src)
image_inverse = cv2.warpPerspective(image_perspective, M_inv, (w, h))
image_inverse_with_artifacts = add_interpolation_artifacts(image_inverse)

# Mostrar las tres imágenes juntas
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image_perspective)
plt.title('Transformación Perspectiva')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image_inverse_with_artifacts)
plt.title('Transformación Inversa ')
plt.axis('off')

plt.tight_layout()
plt.show()
