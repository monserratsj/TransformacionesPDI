import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
img = cv2.imread('chop.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Tamaño de la malla
rows, cols = 10, 10

# Crear un grid de puntos
height, width, _ = img.shape
x = np.linspace(0, width, cols)
y = np.linspace(0, height, rows)
xv, yv = np.meshgrid(x, y)

# Convertir el grid en una lista de puntos
original_points = np.array([xv.flatten(), yv.flatten()]).T
transformed_points = original_points.copy()

# Función para dibujar la imagen con el grid
def draw_image_with_grid(img, points, rows, cols):
    img_copy = img.copy()
    for i in range(rows):
        for j in range(cols):
            cv2.circle(img_copy, (int(points[i*cols + j][0]), int(points[i*cols + j][1])), 3, (0, 255, 0), -1)
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

    for i in range(rows-1):
        for j in range(cols-1):
            src_quad = np.array([src_points[i*cols + j], src_points[i*cols + j + 1], 
                                 src_points[(i+1)*cols + j + 1], src_points[(i+1)*cols + j]], np.float32)
            dst_quad = np.array([dst_points[i*cols + j], dst_points[i*cols + j + 1], 
                                 dst_points[(i+1)*cols + j + 1], dst_points[(i+1)*cols + j]], np.float32)
            transform_matrix = cv2.getPerspectiveTransform(src_quad, dst_quad)

            for y in range(int(src_quad[0][1]), int(src_quad[2][1])):
                for x in range(int(src_quad[0][0]), int(src_quad[2][0])):
                    src_pt = np.array([x, y], dtype=np.float32)
                    dst_pt = cv2.perspectiveTransform(src_pt.reshape(1, 1, 2), transform_matrix)
                    map_x[y, x] = dst_pt[0, 0, 0]
                    map_y[y, x] = dst_pt[0, 0, 1]

    transformed_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return transformed_img

# Mostrar la imagen con el grid
img_with_grid = draw_image_with_grid(img, transformed_points, rows, cols)
plt.imshow(img_with_grid)
plt.show()

# Variables globales para el manejo de eventos del mouse
selected_point = None

def find_closest_point(x, y, points):
    closest_dist = float('inf')
    closest_point = None
    for i, (px, py) in enumerate(points):
        dist = np.sqrt((px - x)**2 + (py - y)**2)
        if dist < closest_dist:
            closest_dist = dist
            closest_point = i
    return closest_point

# Función del mouse para mover los puntos
def mouse_callback(event, x, y, flags, param):
    global selected_point, transformed_points
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = find_closest_point(x, y, transformed_points)
    elif event == cv2.EVENT_MOUSEMOVE and selected_point is not None:
        transformed_points[selected_point] = (x, y)
        img_with_grid = draw_image_with_grid(img, transformed_points, rows, cols)
        transformed_img = apply_transformation(img, original_points, transformed_points, rows, cols)
        cv2.imshow('Image with Grid', cv2.cvtColor(img_with_grid, cv2.COLOR_RGB2BGR))
        cv2.imshow('Transformed Image', cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR))
    elif event == cv2.EVENT_LBUTTONUP:
        selected_point = None

# Mostrar la imagen en una ventana de OpenCV
cv2.imshow('Image with Grid', cv2.cvtColor(img_with_grid, cv2.COLOR_RGB2BGR))
cv2.imshow('Transformed Image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
cv2.setMouseCallback('Image with Grid', mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()
