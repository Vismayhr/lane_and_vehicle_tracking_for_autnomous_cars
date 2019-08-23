import numpy as np
import cv2

def birds_eye_view(image):
    (h, w) = (image.shape[0], image.shape[1])
    
    # Source points of original image - hardcoded
    source = np.float32([[w // 2 - 76, h * .625], [w // 2 + 76, h * .625], [-100, h], [w + 100, h]])
    
    # Destination points of bird's eye perspective - hardcoded
    destination = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])
    
    transform_matrix = cv2.getPerspectiveTransform(source, destination)
    unwarp_matrix = cv2.getPerspectiveTransform(destination, source)
    
    # Return the warped image and a way to obtain the original image from the transformed image i.e unwarp_matrix
    return (cv2.warpPerspective(image, transform_matrix, (w, h)), unwarp_matrix)