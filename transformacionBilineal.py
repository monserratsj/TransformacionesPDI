import cv2
import numpy as np

def apply_bilinear_transformation(val):
    
    a0 = cv2.getTrackbarPos('a0', 'Transformada') / 100
    a1 = cv2.getTrackbarPos('a1', 'Transformada') / 100
    a2 = cv2.getTrackbarPos('a2', 'Transformada') / 100
    a3 = cv2.getTrackbarPos('a3', 'Transformada') / 100 * 0.0001  # Más sutil debido al producto xy
    b0 = cv2.getTrackbarPos('b0', 'Transformada') / 100
    b1 = cv2.getTrackbarPos('b1', 'Transformada') / 100
    b2 = cv2.getTrackbarPos('b2', 'Transformada') / 100
    b3 = cv2.getTrackbarPos('b3', 'Transformada') / 100 * 0.0001  # Más sutil debido al producto xy

    new_image = np.zeros_like(image)

    # Aplicar la transformación bilineal
    for y in range(height):
        for x in range(width):
            new_x = int(a0 + a1*x + a2*y + a3*x*y) % width
            new_y = int(b0 + b1*x + b2*y + b3*x*y) % height
            new_image[new_y, new_x] = image[y, x]

   
    cv2.imshow('Transformada', new_image)

image_path = "C:\\Users\\Roberto Zamora\\Downloads\\mouse.jpg"
image = cv2.imread(image_path)
if image is None:
    print("Error al leer la imagen.")
else:
    height, width = image.shape[:2]

    cv2.namedWindow('Transformada')

    # Sliders para ajustar los coeficientes bilineales
    cv2.createTrackbar('a0', 'Transformada', 100, 200, apply_bilinear_transformation)
    cv2.createTrackbar('a1', 'Transformada', 100, 200, apply_bilinear_transformation)
    cv2.createTrackbar('a2', 'Transformada', 100, 200, apply_bilinear_transformation)
    cv2.createTrackbar('a3', 'Transformada', 100, 200, apply_bilinear_transformation)
    cv2.createTrackbar('b0', 'Transformada', 100, 200, apply_bilinear_transformation)
    cv2.createTrackbar('b1', 'Transformada', 100, 200, apply_bilinear_transformation)
    cv2.createTrackbar('b2', 'Transformada', 100, 200, apply_bilinear_transformation)
    cv2.createTrackbar('b3', 'Transformada', 100, 200, apply_bilinear_transformation)

    apply_bilinear_transformation(0)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
