import cv2
import numpy as np

def update_perspective(val):
    # Puntos deseados en la nueva imagen ajustados con los valores de los sliders
    dst_points = np.float32([
        [cv2.getTrackbarPos('X0', 'Transformada') / 100 * width, cv2.getTrackbarPos('Y0', 'Transformada') / 100 * height],
        [cv2.getTrackbarPos('X1', 'Transformada') / 100 * width, cv2.getTrackbarPos('Y1', 'Transformada') / 100 * height],
        [cv2.getTrackbarPos('X2', 'Transformada') / 100 * width, cv2.getTrackbarPos('Y2', 'Transformada') / 100 * height],
        [cv2.getTrackbarPos('X3', 'Transformada') / 100 * width, cv2.getTrackbarPos('Y3', 'Transformada') / 100 * height]
    ])
    
    # Matriz de transformación de perspectiva
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Aplicar la transformación
    transformed_image = cv2.warpPerspective(image, matrix, (width, height))
    
    # Mostrar la imagen transformada
    cv2.imshow('Transformada', transformed_image)

# Ruta de la imagen a transformar
image_path = 'oso.jpg'
image = cv2.imread(image_path)
if image is None:
    print("Error al leer la imagen.")
else:
    # Dimensiones de la imagen
    height, width = image.shape[:2]
    
    # Puntos de la imagen original
    src_points = np.float32([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]])
    
    # Crear una ventana para mostrar la imagen transformada
    cv2.namedWindow('Transformada')
    
    # Crear sliders para ajustar la perspectiva
    cv2.createTrackbar('X0', 'Transformada', 0, 100, update_perspective)
    cv2.createTrackbar('Y0', 'Transformada', 0, 100, update_perspective)
    cv2.createTrackbar('X1', 'Transformada', 100, 100, update_perspective)
    cv2.createTrackbar('Y1', 'Transformada', 0, 100, update_perspective)
    cv2.createTrackbar('X2', 'Transformada', 0, 100, update_perspective)
    cv2.createTrackbar('Y2', 'Transformada', 100, 100, update_perspective)
    cv2.createTrackbar('X3', 'Transformada', 100, 100, update_perspective)
    cv2.createTrackbar('Y3', 'Transformada', 100, 100, update_perspective)
    
    # Mostrar la imagen original y la transformada por primera vez
    update_perspective(0)
    
    # Esperar a que el usuario presione una tecla para salir
    cv2.waitKey(0)
    cv2.destroyAllWindows()
