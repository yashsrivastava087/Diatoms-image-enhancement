import cv2
import numpy as np

# --- METRIC FUNCTIONS ---
def compute_entropy(img):
    """Shannon entropy of a grayscale image."""
    hist = cv2.calcHist([img], [0], None, [256], [0,256]).flatten()
    p = hist / hist.sum()
    p = p[p>0]
    return -(p * np.log2(p)).sum()

def compute_average_gradient(img):
    """Mean gradient magnitude via Sobel."""
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    return grad.mean()

def compute_uiqi(ref, targ):
    """Universal Image Quality Index between two grayscales."""
    ref = ref.astype(np.float64); targ = targ.astype(np.float64)
    μx, μy = ref.mean(), targ.mean()
    σx2 = ((ref-μx)**2).mean(); σy2 = ((targ-μy)**2).mean()
    σxy = ((ref-μx)*(targ-μy)).mean()
    return (4 * σxy * μx * μy) / ((σx2 + σy2) * (μx**2 + μy**2))

def compute_psnr(ref, targ):
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((ref.astype(np.float64) - targ.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10((255.0**2) / mse)

def compute_ebcm(img):
    """Edge-Based Contrast Measure: mean gradient on Canny edges."""
    edges = cv2.Canny(img, 100, 200)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    return grad[edges>0].mean() if edges.any() else 0

# --- LOAD ---
img = cv2.imread('diatomic_dataset/train/image.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- ENHANCEMENT (CLAHE + UN­SHARP + GAMMA) ---
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
l = clahe.apply(l)
enh_lab = cv2.merge((l, a, b))
enhanced_img = cv2.cvtColor(enh_lab, cv2.COLOR_LAB2BGR)
gauss = cv2.GaussianBlur(enhanced_img, (0,0), 3)
sharpened = cv2.addWeighted(enhanced_img, 1.5, gauss, -0.5, 0)
gamma = 1.2
lut = np.array([((i/255.0)**gamma)*255 for i in range(256)], dtype='uint8')
final_enhanced = cv2.LUT(sharpened, lut)
enh_gray = cv2.cvtColor(final_enhanced, cv2.COLOR_BGR2GRAY)

# --- BACKGROUND REMOVAL (GrabCut) ---
mask = np.zeros(img.shape[:2], np.uint8)
bgM = np.zeros((1,65), np.float64)
fgM = np.zeros((1,65), np.float64)
h, w = img.shape[:2]
cv2.grabCut(img, mask, (10,10,w-20,h-20), bgM, fgM, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
background_removed = img * mask2[:,:,None]
bg_gray = cv2.cvtColor(background_removed, cv2.COLOR_BGR2GRAY)

# --- COMPUTE & PRINT METRICS ---
datasets = {
    'Original':        gray,
    'Enhanced':        enh_gray,
    'BackgroundRemoved': bg_gray
}

print("Metrics (Entropy, AvgGrad, UIQI, PSNR, EBCM):")
for name, g in datasets.items():
    ent = compute_entropy(g)
    ag  = compute_average_gradient(g)
    eb  = compute_ebcm(g)
    if name == 'Original':
        uiqi = psnr = None
    else:
        uiqi = compute_uiqi(gray, g)
        psnr = compute_psnr(gray, g)
    print(f"{name:20s}"
          f"  Ent={ent:6.3f}"
          f"  AvgG={ag:6.3f}"
          f"  EBCM={eb:6.3f}"
          + (f"  UIQI={uiqi:5.3f}  PSNR={psnr:5.2f}" if uiqi is not None else "")
         )

# --- DISPLAY IN BGR ---
cv2.imshow('Original Input', img)
cv2.imshow('Enhanced Image', final_enhanced)
cv2.imshow('Background Removed Image', background_removed)
cv2.waitKey(0)
cv2.destroyAllWindows()
