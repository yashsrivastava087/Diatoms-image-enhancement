import cv2

img = cv2.imread('diatomic_dataset/train/image3.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

result = cv2.bitwise_and(img, img, mask=mask)

enhanced_img = cv2.convertScaleAbs(result, alpha=1.5,beta=0)

cv2.imshow('Original', img)
cv2.imshow('Background Removed', result)  
cv2.imshow('Enhanced', enhanced_img)
cv2.waitKey(0)
cv2.destroyAllWindows()