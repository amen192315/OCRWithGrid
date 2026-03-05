#pylint: disable=no-member
import cv2
import numpy as np
from PIL import Image
import pytesseract
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
FILE_NAME = r'D:\pykyrs\englishPhotojpg.jpg'

img_original = cv2.imread(FILE_NAME)
if img_original is None:
    print(f"Error: failed to load {FILE_NAME}")
    sys.exit(1)

h_orig, w_orig = img_original.shape[:2]
img_current = img_original.copy()
img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
h, w = h_orig, w_orig
angle = 0

GRID_STEP = 80
grid_points = []
selection_rect = None
selected_point = None

def create_grid_points():
    points = []
    for i in range(0, h, GRID_STEP):
        for j in range(0, w, GRID_STEP):
            points.append((j, i))
    return points

def draw_grid(img, highlight_point=None):
    grid_img = img.copy()
    
    for i in range(0, h, GRID_STEP):
        cv2.line(grid_img, (0, i), (w, i), (0, 255, 0), 2)
    for j in range(0, w, GRID_STEP):
        cv2.line(grid_img, (j, 0), (j, h), (0, 255, 0), 2)
    
    for point in grid_points:
        color = (0, 255, 255) if point == highlight_point else (0, 255, 0)
        cv2.circle(grid_img, point, 15, color, -1)
    
    if selection_rect:
        x, y, rw, rh = selection_rect
        cv2.rectangle(grid_img, (x, y), (x+rw, y+rh), (0, 0, 255), 3)
    
    return grid_img

def rotate_region_around_point(center, angle_deg):
    global img_current, img_gray
    
    cx, cy = center
    size = GRID_STEP
    half = size // 2
    
    x1, y1 = max(0, cx-half), max(0, cy-half)
    x2, y2 = min(w, cx+half), min(h, cy+half)
    
    if x2 <= x1 or y2 <= y1:
        return
        
    region = img_current[y1:y2, x1:x2].copy()
    rows, cols = region.shape[:2]
    
    M = cv2.getRotationMatrix2D((cols//2, rows//2), angle_deg, 1.0)
    rotated = cv2.warpAffine(region, M, (cols, rows), 
                            flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_REPLICATE) 
    
    img_current[y1:y2, x1:x2] = rotated
    img_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
    print(f"CUBE ROTATED {center} by {angle_deg}°")

def global_rotate(angle_deg):
    global img_current, img_gray, angle   
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle + angle_deg, 1.0)
    img_current = cv2.warpAffine(img_original, M, (w, h), 
                                flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0)) 
    img_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
    angle += angle_deg
    print(f"GLOBAL ROTATION {angle:.1f}° - NO MIRRORING!")

def mouse_callback(event, x, y, flags, param):
    global selected_point, selection_rect
    
    if event == cv2.EVENT_LBUTTONDOWN:
        min_dist = float('inf')
        closest = None
        for point in grid_points:
            dist = ((point[0]-x)**2 + (point[1]-y)**2)**0.5
            if dist < 25 and dist < min_dist:
                min_dist = dist
                closest = point
        
        if closest:
            selected_point = closest
            half = GRID_STEP // 2
            selection_rect = (closest[0]-half, closest[1]-half, GRID_STEP, GRID_STEP)
            print(f"CUBE SELECTED: {closest}")
        else:
            selected_point = None
            selection_rect = None
            print("No cube selected")
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        selected_point = None
        selection_rect = None
        print("FOCUS CLEARED!")

def ocr_process():
    gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    cv2.imwrite('processed.jpg', thresh)
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(Image.fromarray(thresh), lang='eng', config=config)
    
    with open('ocr_result.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    
    return text, thresh

grid_points = create_grid_points()
cv2.namedWindow('Grid Tool', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Grid Tool', mouse_callback)

while True:
    grid_points = create_grid_points()
    
    display = draw_grid(img_current, selected_point)
    
    status = "CUBE ACTIVE" if selected_point else "FREE MODE"
    cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(display, f"Angle: {angle:.1f}°", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow('Grid Tool', display)
    
    key = cv2.waitKey(20) & 0xFF
    
    if selected_point is not None:
        if key == ord('a') or key == ord('A'):
            rotate_region_around_point(selected_point, -10)
        elif key == ord('d') or key == ord('D'):
            rotate_region_around_point(selected_point, 10)
    
    elif key == ord('a') or key == ord('A'):
        global_rotate(-10)
    elif key == ord('d') or key == ord('D'):
        global_rotate(10)
    
    elif key == 13:  # ENTER
        print("ENTER: Saving final.jpg...")
        cv2.imwrite('final.jpg', img_current)
        text, thresh = ocr_process()
        print("OCR result:", repr(text[:200]))
        cv2.imshow('OCR Preview', thresh)
        cv2.waitKey(0)
        cv2.destroyWindow('OCR Preview')
    
    elif key == ord('s') or key == ord('S'):  # S
        text, thresh = ocr_process()
        print("OCR result:", repr(text[:200]))
        cv2.imshow('OCR Preview', thresh)
        cv2.waitKey(0)
        cv2.destroyWindow('OCR Preview')
    
    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()
