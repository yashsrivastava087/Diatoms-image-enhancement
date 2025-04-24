import cv2
import numpy as np

# --- METRIC FUNCTIONS ---
def compute_entropy(img):
    """Shannon entropy H = -∑ p_i log2(p_i) over normalized histogram."""
    hist = cv2.calcHist([img], [0], None, [256], [0,256]).flatten()
    p = hist / hist.sum()
    p = p[p>0]
    return -(p * np.log2(p)).sum()

def compute_average_gradient(img):
    """Mean gradient magnitude via Sobel filters."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    return grad.mean()

def compute_uiqi(ref, targ):
    """Universal Image Quality Index between two images."""
    ref  = ref.astype(np.float64)
    targ = targ.astype(np.float64)
    μx = ref.mean()
    μy = targ.mean()
    σx2 = ((ref - μx)**2).mean()
    σy2 = ((targ - μy)**2).mean()
    σxy = ((ref - μx)*(targ - μy)).mean()
    return (4 * σxy * μx * μy) / ((σx2 + σy2) * (μx**2 + μy**2))

def compute_psnr(ref, targ):
    """Peak Signal‑to‑Noise Ratio between two images."""
    mse = np.mean((ref.astype(np.float64) - targ.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 10 * np.log10((PIXEL_MAX**2) / mse)

def compute_ebcm(img):
    
    edges = cv2.Canny(img, 100, 200)
    # gradient magnitude
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    # average only on edge pixels
    if edges.sum() == 0:
        return 0
    return grad[edges>0].mean()

# --- LOAD & PROCESS IMAGE ---
img = cv2.imread('diatomic_dataset/train/image.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Enhancement (IABR/IABR‑style normalization)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
l = ((l - l.min()) / (l.max() - l.min()) * 255).astype(np.uint8)
l_blur = cv2.GaussianBlur(l, (5, 5), 0)
l = cv2.addWeighted(l, 1.5, l_blur, -0.5, 0)
enh_lab = cv2.merge((l, a, b))
enh = cv2.cvtColor(enh_lab, cv2.COLOR_LAB2BGR)
gauss = cv2.GaussianBlur(enh, (0, 0), 3)
sharp = cv2.addWeighted(enh, 1.5, gauss, -0.5, 0)
gamma = 1.2
lut = np.array([((i/255.)**gamma)*255 for i in range(256)], dtype='uint8')
enhanced = cv2.LUT(sharp, lut)
enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

# Background removal (GrabCut)
mask = np.zeros(img.shape[:2], np.uint8)
bgM = np.zeros((1,65), np.float64)
fgM = np.zeros((1,65), np.float64)
h, w = img.shape[:2]
cv2.grabCut(img, mask, (10,10,w-20,h-20), bgM, fgM, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
bgr_removed = img * mask2[:,:,None]
bg_gray = cv2.cvtColor(bgr_removed, cv2.COLOR_BGR2GRAY)

# --- COMPUTE METRICS ---
metrics = {
    'Original': {
        'Entropy':     compute_entropy(gray),
        'Avg Gradient':compute_average_gradient(gray),
        'EBCM':        compute_ebcm(gray),
    },
    'Enhanced': {
        'Entropy':     compute_entropy(enh_gray),
        'Avg Gradient':compute_average_gradient(enh_gray),
        'EBCM':        compute_ebcm(enh_gray),
        'UIQI':        compute_uiqi(gray, enh_gray),
        'PSNR':        compute_psnr(gray, enh_gray),
    },
    'Background Removed': {
        'Entropy':     compute_entropy(bg_gray),
        'Avg Gradient':compute_average_gradient(bg_gray),
        'EBCM':        compute_ebcm(bg_gray),
        'UIQI':        compute_uiqi(gray, bg_gray),
        'PSNR':        compute_psnr(gray, bg_gray),
    }
}

# --- DISPLAY RESULTS ---
print("Quantitative Metrics (as in Table I) :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}:")
for name, m in metrics.items():
    print(f"\n{name}:")
    for k, v in m.items():
        print(f"  {k:15s}: {v:.4f}")

# --- SHOW IMAGES ---


# --- DISPLAY RESULTS IN BGR COLOR ---
cv2.imshow('Original (color)',        img)           # original BGR
cv2.imshow('Enhanced (color)',        enhanced)      # enhanced BGR
cv2.imshow('Background Removed (color)', bgr_removed) # background‑removed BGR

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.waitKey(0)
cv2.destroyAllWindows()