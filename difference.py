import cv2
import numpy as np

# --- METRIC FUNCTIONS ---
def compute_entropy(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256]).flatten()
    p = hist / hist.sum()
    p = p[p>0]
    return -(p*np.log2(p)).sum()

def compute_average_gradient(img):
    gx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    gy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    grad = np.sqrt(gx*gx + gy*gy)
    return grad.mean()

def compute_uiqi(ref, targ):
    ref = ref.astype(np.float64); targ = targ.astype(np.float64)
    μx, μy = ref.mean(), targ.mean()
    σx2 = ((ref-μx)**2).mean()
    σy2 = ((targ-μy)**2).mean()
    σxy = ((ref-μx)*(targ-μy)).mean()
    return (4*σxy*μx*μy) / ((σx2+σy2)*(μx*μx+μy*μy))

def compute_psnr(ref, targ):
    mse = np.mean((ref.astype(np.float64)-targ.astype(np.float64))**2)
    if mse == 0: return float('inf')
    return 10 * np.log10((255.0**2) / mse)

def compute_ebcm(img):
    edges = cv2.Canny(img,100,200)
    gx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    gy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    grad = np.sqrt(gx*gx + gy*gy)
    return grad[edges>0].mean() if edges.any() else 0

# --- BACKGROUND REMOVAL (GrabCut) ---
def grabcut_cell(bgr):
    mask = np.zeros(bgr.shape[:2], np.uint8)
    bgM = np.zeros((1,65), np.float64)
    fgM = np.zeros((1,65), np.float64)
    h,w = bgr.shape[:2]
    cv2.grabCut(bgr, mask, (10,10,w-20,h-20), bgM, fgM, 5, cv2.GC_INIT_WITH_RECT)
    m2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    cell = bgr * m2[:,:,None]
    return cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

# --- PIPELINES ---
def clahe_pipeline(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2,a,b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    gauss = cv2.GaussianBlur(enhanced, (0,0), 3)
    sharp = cv2.addWeighted(enhanced,1.5,gauss,-0.5,0)
    gamma = 1.2
    lut = np.array([((i/255.0)**gamma)*255 for i in range(256)],dtype='uint8')
    final = cv2.LUT(sharp, lut)
    return final

def iabr_pipeline(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l = ((l-l.min())/(l.max()-l.min())*255).astype('uint8')
    blur = cv2.GaussianBlur(l, (5,5), 0)
    l2 = cv2.addWeighted(l,1.5,blur,-0.5,0)
    merged = cv2.merge((l2,a,b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    gauss = cv2.GaussianBlur(enhanced,(0,0),3)
    sharp = cv2.addWeighted(enhanced,1.5,gauss,-0.5,0)
    gamma = 1.2
    lut = np.array([((i/255.0)**gamma)*255 for i in range(256)],dtype='uint8')
    final = cv2.LUT(sharp, lut)
    return final

# --- MAIN ---
img = cv2.imread('diatomic_dataset/train/image.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Run both pipelines
clahe_bgr = clahe_pipeline(img)
iabr_bgr  = iabr_pipeline(img)

# Extract cell only  
clahe_cell = grabcut_cell(clahe_bgr)
iabr_cell  = grabcut_cell(iabr_bgr)

# Compute metrics
results = {}
for name, cell in [('CLAHE',clahe_cell), ('IABR',iabr_cell)]:
    ent  = compute_entropy(cell)
    ag   = compute_average_gradient(cell)
    eb   = compute_ebcm(cell)
    uiqi = compute_uiqi(gray, cell)
    psnr = compute_psnr(gray, cell)
    results[name] = (ent, ag, uiqi, psnr, eb)

# Print comparison
print(f"{'Method':6s}  Ent    AvgG   UIQI   PSNR    EBCM")
for m, (ent,ag,ui,ps,eb) in results.items():
    print(f"{m:6s}  {ent:6.3f}  {ag:6.3f}  {ui:6.3f}  {ps:6.2f}  {eb:6.3f}")

# Decide which is better
better = max(results.items(), key=lambda kv: sum(kv[1]))[0]

# Explain why
print([better])

# --- DISPLAY COLOR OUTPUTS ---
cv2.imshow('Original', img)
cv2.imshow('CLAHE Cell', clahe_bgr)
cv2.imshow('IABR Cell',  iabr_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
