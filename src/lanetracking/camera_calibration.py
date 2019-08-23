import numpy as np
import matplotlib.image as mpimg
import cv2

class CameraCalibration:
    def __init__(self, calibration_images, pattern_size=(9,6), retain_calibration_images=False):
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.calibration_images_success = []
        self.calibration_images_error = []
        self.calculate_calibration(calibration_images, pattern_size, retain_calibration_images)
        
    def calibrate_image(self, image):
        if self.camera_matrix is not None and self.distortion_coefficients is not None:
            return cv2.undistort(image, self.camera_matrix, self.distortion_coefficients, 
                                 None, self.camera_matrix)
        else:
            return image
    
    def calculate_calibration(self, calibration_images, pattern_size, retain_calibration_images):
        # Since the object points for all 9x6 chessboards are the same (as they are all images of the same 9x6 
        # chess board), the object points in 3D are the same. Hence, prepare the object_points to represent the 
        # 54 sqaures as (x,y,z): (0,0,0), (1,0,0).... (8,5,0). z is always 0 because all squares on the chessboard
        # are on the same plane.
        pattern = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
        pattern[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
        
        object_points = []    # Depicts 3D points in the real-world space
        image_points = []     # Depicts 2D points in the image's plane
        image_size = None
        
        for i, path in enumerate(calibration_images):
            image = mpimg.imread(path)
            grayscale_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            found, corners = cv2.findChessboardCorners(grayscale_img, pattern_size, None)
            if found:
                object_points.append(pattern)
                image_points.append(corners)
                image_size = (image.shape[1], image.shape[0])
                if retain_calibration_images:
                    cv2.drawChessboardCorners(image, pattern_size, corners, True)
                    self.calibration_images_success.append(image)
            else:
                if retain_calibration_images:
                    self.calibration_images_error.append(image)
                    
        if object_points and image_points:
            _,self.camera_matrix, self.distortion_coefficients, _, _ = cv2.calibrateCamera(
                object_points, image_points, image_size, None, None)