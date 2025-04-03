import cv2
import numpy as np

img = cv2.imread('diatomic_dataset/train/image3.png')

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l = clahe.apply(l)

enhanced_lab = cv2.merge((l, a, b))
enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

gaussian = cv2.GaussianBlur(enhanced_img, (0, 0), 3)
sharpened = cv2.addWeighted(enhanced_img, 1.5, gaussian, -0.5, 0)

# Apply Gamma Correction for brightness balance
gamma = 1.2  
gamma_correction = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
final_img = cv2.LUT(sharpened, gamma_correction)

# Show images
cv2.imshow('Original', img)
cv2.imshow('Enhanced', final_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
