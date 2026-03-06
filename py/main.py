import cv2
import numpy as np
from PIL import Image
import pytesseract
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'
FILE_NAME = r'C:\Users\user\Desktop\pydiplom\OCRWithGrid\englishPhoto.jpg'

img_original = cv2.imread(FILE_NAME)
if img_original is None:
    print(f"Error: failed to load {FILE_NAME}")
    sys.exit(1)

h_orig, w_orig = img_original.shape[:2]
img_current = img_original.copy()
img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
h, w = h_orig, w_orig
angle = 0

cubes = []
drawing_rect = False
rect_start = (0, 0)
current_mouse_pos = (0, 0)
selected_cube_idx = -1

def create_cube_from_rect(x, y, rw, rh):
    cubes.append([x, y, rw, rh, 0.0])
    print(f"New cube #{len(cubes)-1}: ({x},{y}) {rw}x{rh}")

def rotate_cube(idx, angle_deg):
    global img_current, img_gray
    
    cube = cubes[idx]
    x, y, rw, rh = cube[:4]
    cube[4] += angle_deg
    
    region_orig = img_original[y:y+rh, x:x+rw].copy()
    
    center = (rw // 2, rh // 2)
    M = cv2.getRotationMatrix2D(center, cube[4], 1.0)
    
    rotated = cv2.warpAffine(region_orig, M, (rw, rh), 
                           flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_REPLICATE)
    
    img_current[y:y+rh, x:x+rw] = rotated
    img_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)

def global_rotate(angle_deg):
    global img_current, img_gray, angle    
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle + angle_deg, 1.0)
    img_current = cv2.warpAffine(img_original, M, (w, h), 
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    img_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
    angle += angle_deg

def mouse_callback(event, x, y, flags, param):
    global drawing_rect, rect_start, current_mouse_pos, selected_cube_idx
    
    current_mouse_pos = (x, y)
    
    if event == cv2.EVENT_RBUTTONDOWN:
        drawing_rect = True
        rect_start = (x, y)
        selected_cube_idx = -1
    
    elif event == cv2.EVENT_RBUTTONUP:
        if drawing_rect:
            drawing_rect = False
            x1, y1 = rect_start
            x2, y2 = x, y
            rw = abs(x2 - x1)
            rh = abs(y2 - y1)
            
            if rw > 15 and rh > 15:
                create_cube_from_rect(min(x1,x2), min(y1,y2), rw, rh)
                selected_cube_idx = len(cubes) - 1
    
    elif event == cv2.EVENT_LBUTTONDOWN:
        selected_cube_idx = -1
        for i, cube in enumerate(cubes):
            cx, cy, cw, ch, _ = cube
            if (cx-10 <= x <= cx+cw+10) and (cy-10 <= y <= cy+ch+10):
                selected_cube_idx = i
                break

def draw_overlay(img):
    overlay = img.copy()
    
    if drawing_rect and rect_start[0] > 0:
        x1, y1 = rect_start
        x2, y2 = current_mouse_pos
        cv2.rectangle(overlay, (min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2)), (0, 255, 0), 3)
    
    for i, cube in enumerate(cubes):
        x, y, rw, rh, rot = cube
        color = (0, 0, 255) if i == selected_cube_idx else (255, 0, 0)
        thickness = 4 if i == selected_cube_idx else 2
        
        cv2.rectangle(overlay, (x, y), (x+rw, y+rh), color, thickness)
        cv2.putText(overlay, f"#{i}:{rot:.0f}°", (x+5, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return overlay

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

cv2.namedWindow('Resizable Cube OCR Tool v2', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Resizable Cube OCR Tool v2', mouse_callback)

print("CONTROLS:")
print("  RMB+Drag = New cube")
print("  LMB = Select cube | A/D = Rotate SELECTED")
print("  A/D (no cube) = Global rotate")
print("  S/ENTER = OCR | ESC = Exit")
print("  DEL = Delete SELECTED cube")

while True:
    display = draw_overlay(img_current)
    
    status_parts = [f"Cubes: {len(cubes)}"]
    if selected_cube_idx >= 0:
        cube = cubes[selected_cube_idx]
        status_parts.append(f"#{selected_cube_idx}: {cube[3]}x{cube[2]}")
    status_parts.append(f"Angle: {angle:.1f}°")
    
    status = " | ".join(status_parts)
    color = (0, 0, 255) if selected_cube_idx >= 0 else (0, 255, 255)
    
    cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imshow('Resizable Cube OCR Tool v2', display)
    
    key = cv2.waitKey(20) & 0xFF
    
    if selected_cube_idx >= 0:
        if key == ord('a') or key == ord('A'):
            rotate_cube(selected_cube_idx, -10)
        elif key == ord('d') or key == ord('D'):
            rotate_cube(selected_cube_idx, 10)
        elif key == 8 or key == 46:
            del cubes[selected_cube_idx]
            selected_cube_idx = -1
    elif key == ord('a') or key == ord('A'):
        global_rotate(-10)
    elif key == ord('d') or key == ord('D'):
        global_rotate(10)
    
    elif key == 13 or key == ord('s') or key == ord('S'):
        print("Saving final.jpg...")
        cv2.imwrite('final.jpg', img_current)
        text, thresh = ocr_process()
        print("OCR:", repr(text[:200]))
        cv2.imshow('OCR Preview', thresh)
        cv2.waitKey(0)
        cv2.destroyWindow('OCR Preview')
    
    elif key == 27:
        break

cv2.destroyAllWindows()
