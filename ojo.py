import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
img = cv2.imread('chop.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Tamaño de la malla
rows, cols = 100, 100

# Crear un grid de puntos
height, width, _ = img.shape
x = np.linspace(0, width, cols)
y = np.linspace(0, height, rows)
xv, yv = np.meshgrid(x, y)

# Convertir el grid en una lista de puntos
original_points = np.array([xv.flatten(), yv.flatten()]).T

# Función de deformación ojo de pez
def fisheye_effect(x, y, width, height, k=0.00005):
    cx, cy = width / 2, height / 2  # Centro de la imagen
    new_x = x - cx
    new_y = y - cy
    r = np.sqrt(new_x**2 + new_y**2)
    r_new = r * (1 + k * r**2)
    scale = r_new / r
    new_x = new_x * scale + cx
    new_y = new_y * scale + cy
    return new_x, new_y

# Aplicar la deformación ojo de pez
transformed_points = np.zeros_like(original_points)
transformed_points[:, 0], transformed_points[:, 1] = fisheye_effect(original_points[:, 0], original_points[:, 1], width, height)

# Función para dibujar la imagen con el grid
def draw_image_with_grid(img, points, rows, cols):
    img_copy = img.copy()
    for i in range(rows):
        for j in range(cols):
            cv2.circle(img_copy, (int(points[i*cols + j][0]), int(points[i*cols + j][1])), 1, (0, 255, 0), -1)
            if j < cols - 1:
                cv2.line(img_copy, (int(points[i*cols + j][0]), int(points[i*cols + j + 1][0])),
                         (int(points[i*cols + j][1]), int(points[i*cols + j + 1][1])), (255, 0, 0), 1)
            if i < rows - 1:
                cv2.line(img_copy, (int(points[i*cols + j][0]), int(points[(i+1)*cols + j][0])),
                         (int(points[i*cols + j][1]), int(points[(i+1)*cols + j][1])), (255, 0, 0), 1)
    return img_copy

# Función para transformar la imagen
def apply_transformation(img, src_points, dst_points, rows, cols):
    map_x = np.zeros(img.shape[:2], dtype=np.float32)
    map_y = np.zeros(img.shape[:2], dtype=np.float32)

    src_points = src_points.reshape((rows, cols, 2))
    dst_points = dst_points.reshape((rows, cols, 2))

    for i in range(rows - 1):
        for j in range(cols - 1):
            src_quad = np.array([src_points[i, j], src_points[i, j + 1], src_points[i + 1, j + 1], src_points[i + 1, j]], np.float32)
            dst_quad = np.array([dst_points[i, j], dst_points[i, j + 1], dst_points[i + 1, j + 1], dst_points[i + 1, j]], np.float32)
            transform_matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)

            min_x = int(min(src_quad[:, 0]))
            max_x = int(max(src_quad[:, 0]))
            min_y = int(min(src_quad[:, 1]))
            max_y = int(max(src_quad[:, 1]))

            for y in range(min_y, max_y):
                for x in range(min_x, max_x):
                    src_pt = np.array([[x, y]], dtype=np.float32)
                    dst_pt = cv2.perspectiveTransform(src_pt[None, :, :], transform_matrix)
                    if 0 <= dst_pt[0, 0, 0] < width and 0 <= dst_pt[0, 0, 1] < height:
                        map_x[y, x] = dst_pt[0, 0, 0]
                        map_y[y, x] = dst_pt[0, 0, 1]

    transformed_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return transformed_img

# Mostrar la imagen con el grid
img_with_grid = draw_image_with_grid(img, transformed_points, rows, cols)
plt.imshow(img_with_grid)
plt.show()

# Aplicar la transformación y mostrar la imagen transformada
transformed_img = apply_transformation(img, original_points, transformed_points, rows, cols)
plt.imshow(transformed_img)
plt.show()
