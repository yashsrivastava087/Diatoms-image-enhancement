import cv2
import numpy as np

# Read image
img = cv2.imread('diatomic_dataset/train/image3.png')

# Convert to LAB color space
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# Normalize L-channel using IABR method
l_min, l_max = np.min(l), np.max(l)
l = ((l - l_min) / (l_max - l_min) * 255).astype(np.uint8)

# Apply Gaussian Blur for smoothing
l_blurred = cv2.GaussianBlur(l, (5,5), 0)

# Adjust brightness using weighted addition
enhanced_l = cv2.addWeighted(l, 1.5, l_blurred, -0.5, 0)

# Merge back the enhanced L-channel with A and B
enhanced_lab = cv2.merge((enhanced_l, a, b))
enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

# Apply Unsharp Masking (Sharpening)
gaussian = cv2.GaussianBlur(enhanced_img, (0, 0), 3)
sharpened = cv2.addWeighted(enhanced_img, 1.5, gaussian, -0.5, 0)

# Apply Gamma Correction
gamma = 1.2  # Adjust as needed
gamma_correction = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
final_img = cv2.LUT(sharpened, gamma_correction)

# Show images
cv2.imshow('Original', img)
cv2.imshow('Enhanced (IABR Method)', final_img)

cv2.waitKey(0);
cv2.destroyAllWindows();
