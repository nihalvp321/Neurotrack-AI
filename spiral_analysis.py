import cv2
import numpy as np
import os

IMG_SIZE = 128

def extract_spiral_features(image_path):
    """
    Extracts 29 clinical features from a spiral drawing image.
    Matches the logic used in the training notebook.
    """
    # 1. Load and Preprocess
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # Stronger blur (7,7) to handle digital stroke aliasing more effectively
    img = cv2.GaussianBlur(img, (7, 7), 0)
    image = img / 255.0  # Normalize to 0-1
    
    # Convert back to uint8 for OpenCV operations
    img_uint8 = (image * 255).astype(np.uint8)
    
    # 2. Binarize using Otsu's thresholding for better robustness to light/colored ink
    _, binary = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Edge Detection
    edges = cv2.Canny(img_uint8, 50, 150)
    
    # 4. Find Contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Basic Statistics
    pixel_std = np.std(image)
    pixel_mean = np.mean(image)
    pixel_max = np.max(image)
    
    # 6. Gradient Features
    sobelx = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    grad_mean = np.mean(gradient_magnitude)
    grad_std = np.std(gradient_magnitude)
    grad_max = np.max(gradient_magnitude)
    
    # 7. Laplacian (blur measure)
    laplacian = cv2.Laplacian(img_uint8, cv2.CV_64F)
    laplacian_var = np.var(laplacian)
    laplacian_mean = np.mean(np.abs(laplacian))
    
    # 8. Contour Features
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest)
        contour_perimeter = cv2.arcLength(largest, True)
        
        if contour_perimeter > 0:
            circularity = (4 * np.pi * contour_area / (contour_perimeter ** 2))
        else:
            circularity = 0
            
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        solidity = (contour_area / hull_area if hull_area > 0 else 0)
        
        x, y, w, h = cv2.boundingRect(largest)
        aspect_ratio = w / (h + 1e-10)
        extent = (contour_area / (w * h) if w * h > 0 else 0)
        num_contours = len(contours)
    else:
        contour_area = contour_perimeter = circularity = solidity = aspect_ratio = extent = num_contours = 0
    
    # 9. Texture Features
    diff_h = np.abs(np.diff(image, axis=1))
    diff_v = np.abs(np.diff(image, axis=0))
    texture_h_mean = np.mean(diff_h)
    texture_h_std = np.std(diff_h)
    texture_v_mean = np.mean(diff_v)
    texture_v_std = np.std(diff_v)
    
    # 10. Frequency Features
    fft_rows = np.fft.fft2(image)
    fft_mag = np.abs(np.fft.fftshift(fft_rows))
    center = IMG_SIZE // 2
    radius = IMG_SIZE // 4
    y_grid, x_grid = np.ogrid[-center:IMG_SIZE-center, -center:IMG_SIZE-center]
    mask = x_grid**2 + y_grid**2 > radius**2
    high_freq_ratio = np.sum(fft_mag[mask]) / (np.sum(fft_mag) + 1e-10)
    
    # 11. Edge Density
    edge_density = np.sum(edges > 0) / edges.size
    
    # 12. HOG-like
    angle = np.arctan2(sobely, sobelx + 1e-10)
    hist, _ = np.histogram(angle.flatten(), bins=8, range=(-np.pi, np.pi), weights=gradient_magnitude.flatten())
    hist = hist / (hist.sum() + 1e-10)
    
    # Combine all 29
    features = [
        pixel_std, pixel_mean, pixel_max,
        grad_mean, grad_std, grad_max,
        laplacian_var, laplacian_mean,
        contour_area, contour_perimeter, circularity, solidity, aspect_ratio, extent, num_contours,
        texture_h_mean, texture_h_std, texture_v_mean, texture_v_std,
        high_freq_ratio, edge_density
    ]
    features.extend(hist.tolist())
    
    return np.array(features)
