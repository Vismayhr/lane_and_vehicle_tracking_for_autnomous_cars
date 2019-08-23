import numpy as np
import cv2

def absolute_gradient_mask(image, kernel_size=3, axis='x', threshold = (0,255)):
    if axis=='x':
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size))
    elif axis=='y':
        sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size))
    sobel = np.uint8(255 * sobel / np.amax(sobel))
    mask = np.zeros_like(sobel)
    mask[(sobel >= threshold[0]) & (sobel <= threshold[1])] = 1
    return mask

def gradient_magnitude_mask(image, kernel_size=3, threshold=(0,255)):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.amax(gradient_magnitude))
    mask = np.zeros_like(gradient_magnitude)
    mask[(gradient_magnitude >= threshold[0]) & (gradient_magnitude <= threshold[1])] = 1
    return mask

def gradient_direction_mask(image, kernel_size=3, threshold=(0, np.pi / 2)):
    sobel_x_abs = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size))
    sobel_y_abs = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size))
    gradient_direction = np.arctan2(sobel_y_abs, sobel_x_abs)
    mask = np.zeros_like(gradient_direction)
    mask[(gradient_direction >= threshold[0]) & (gradient_direction <= threshold[1])] = 1
    return mask

def color_threshold_mask(image, threshold=(0,255)):
    mask = np.zeros_like(image)
    mask[(image >= threshold[0]) & (image <= threshold[1])] = 1
    return mask

def get_edges(image, seperate_channels=False):
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    saturation_channel = hls_image[:,:,2]
    
    absolute_gradient_x = absolute_gradient_mask(saturation_channel, kernel_size=3, axis='x', threshold=(20,100))
    absolute_gradient_y = absolute_gradient_mask(saturation_channel, kernel_size=3, axis='y', threshold=(20,100))
    gradient_magnitude = gradient_magnitude_mask(saturation_channel, kernel_size=3, threshold=(20,100))
    gradient_direction = gradient_direction_mask(saturation_channel, kernel_size=3, threshold=(0.7,1.3))
    
    gradient_mask = np.zeros_like(saturation_channel)
    gradient_mask[((absolute_gradient_x==1) & (absolute_gradient_y==1)) | 
                  ((gradient_magnitude==1) & (gradient_direction==1))] = 1
    
    color_mask = color_threshold_mask(saturation_channel, threshold=(170,255))
    
    if seperate_channels:
        return np.dstack((np.zeros_like(saturation_channel), gradient_mask, color_mask))
    else:
        mask = np.zeros_like(gradient_mask)
        mask[(gradient_mask==1) | (color_mask==1)] = 1
        return mask
