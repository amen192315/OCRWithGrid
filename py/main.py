import cv2
import numpy as np
from PIL import Image
import pytesseract
import sys
import io
import time

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

# Панель управления
PANEL_HEIGHT = 80
BUTTON_HEIGHT = 60
buttons = {
    'rotate_left': {'x': 10, 'y': 10, 'w': 80, 'h': BUTTON_HEIGHT, 'text': '<-10°'},
    'rotate_right': {'x': 100, 'y': 10, 'w': 80, 'h': BUTTON_HEIGHT, 'text': '10°->'},
    'delete': {'x': 190, 'y': 10, 'w': 80, 'h': BUTTON_HEIGHT, 'text': 'Delete'},
    'ocr': {'x': 280, 'y': 10, 'w': 80, 'h': BUTTON_HEIGHT, 'text': 'OCR'},
    'reset': {'x': 370, 'y': 10, 'w': 80, 'h': BUTTON_HEIGHT, 'text': 'Reset'}
}

# HOLD TO ROTATE
button_pressed = None
last_rotate_time = 0
ROTATE_INTERVAL = 0.1

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

def reset_all():
    global cubes, selected_cube_idx, img_current, img_gray, angle, button_pressed
    cubes.clear()
    selected_cube_idx = -1
    img_current = img_original.copy()
    img_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
    angle = 0
    button_pressed = None
    print("Reset all!")

def mouse_callback(event, x, y, flags, param):
    global drawing_rect, rect_start, current_mouse_pos, selected_cube_idx, button_pressed, last_rotate_time
    
    current_time = time.time()
    
    # ✅ ОБРАБОТКА КНОПОК ПРИ КЛИКЕ
    if event == cv2.EVENT_LBUTTONDOWN and y >= h_orig:
        panel_y = y - h_orig
        clicked_button = None
        for btn_name, btn in buttons.items():
            if (btn['x'] <= x <= btn['x'] + btn['w'] and 
                btn['y'] <= panel_y <= btn['y'] + btn['h']):
                clicked_button = btn_name
                break
        
        if clicked_button:
            print(f"Clicked: {clicked_button}")  # DEBUG
            
            button_pressed = clicked_button if clicked_button in ['rotate_left', 'rotate_right'] else None
            
            # ❌ НЕ РОТАЦИОННЫЕ КНОПКИ - ВЫПОЛНЯЕМ СРАЗУ
            if clicked_button == 'delete' and selected_cube_idx >= 0:
                del cubes[selected_cube_idx]
                selected_cube_idx = -1
                print("Cube deleted!")
            elif clicked_button == 'ocr':
                print("Saving final.jpg...")
                cv2.imwrite('final.jpg', img_current)
                ocr_process()
            elif clicked_button == 'reset':
                reset_all()
            elif clicked_button in ['rotate_left', 'rotate_right']:
                # Первый поворот для ротации
                last_rotate_time = current_time
                angle_step = -10 if clicked_button == 'rotate_left' else 10
                if selected_cube_idx >= 0:
                    rotate_cube(selected_cube_idx, angle_step)
                else:
                    global_rotate(angle_step)
        return
    
    # ✅ ОСВОБОЖДЕНИЕ КНОПКИ
    elif event == cv2.EVENT_LBUTTONUP:
        button_pressed = None
        return
    
    # ✅ НЕПРЕРЫВНАЯ РОТАЦИЯ при зажатии
    elif event == cv2.EVENT_MOUSEMOVE and button_pressed and (flags & cv2.EVENT_FLAG_LBUTTON):
        if button_pressed in ['rotate_left', 'rotate_right'] and current_time - last_rotate_time >= ROTATE_INTERVAL:
            angle_step = -10 if button_pressed == 'rotate_left' else 10
            if selected_cube_idx >= 0:
                rotate_cube(selected_cube_idx, angle_step)
            else:
                global_rotate(angle_step)
            last_rotate_time = current_time
        return
    
    # ✅ ЛКМ НА ИЗОБРАЖЕНИИ (выделение куба)
    if event == cv2.EVENT_LBUTTONDOWN and y < h_orig:
        selected_cube_idx = -1
        for i, cube in enumerate(cubes):
            cx, cy, cw, ch, _ = cube
            if (cx-10 <= x <= cx+cw+10) and (cy-10 <= y <= cy+ch+10):
                selected_cube_idx = i
                break
        return
    
    # ✅ РИСУНОК КУБА (ПРАВАЯ КНОПКА)
    if y >= h_orig or button_pressed:
        return
        
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

def draw_combined_image(img):
    combined = np.zeros((h_orig + PANEL_HEIGHT, w_orig, 3), dtype=np.uint8)
    combined[:h_orig, :, :] = img
    
    # Статус сверху
    status_parts = [f"Cubes: {len(cubes)}"]
    if selected_cube_idx >= 0:
        cube = cubes[selected_cube_idx]
        status_parts.append(f"#{selected_cube_idx}: {cube[2]}x{cube[3]}")
    status_parts.append(f"Angle: {angle:.1f}°")
    status = " | ".join(status_parts)
    
    cv2.rectangle(combined, (5, 5), (w_orig-5, 35), (0, 0, 0), -1)
    cv2.putText(combined, status, (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Панель
    panel_y = h_orig
    cv2.rectangle(combined, (0, panel_y), (w_orig, h_orig + PANEL_HEIGHT), (50, 50, 50), -1)
    
    # Кнопки
    for btn_name, btn in buttons.items():
        btn_y = panel_y + btn['y']
        color = (120, 180, 230) if button_pressed == btn_name else (100, 150, 200) if btn_name == 'ocr' else (70, 70, 70)
        
        cv2.rectangle(combined, (btn['x'], btn_y), 
                     (btn['x']+btn['w'], btn_y+btn['h']), color, -1)
        cv2.rectangle(combined, (btn['x'], btn_y), 
                     (btn['x']+btn['w'], btn_y+btn['h']), (255,255,255), 2)
        
        text_size = cv2.getTextSize(btn['text'], cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)[0]
        text_x = btn['x'] + (btn['w'] - text_size[0]) // 2
        text_y = btn_y + ((btn['h'] - text_size[1]) // 2) + 12
        cv2.putText(combined, btn['text'], (int(text_x), int(text_y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 2)
    
    return combined

def draw_overlay(img):
    overlay = img.copy()
    
    if drawing_rect and rect_start[0] > 0:
        x1, y1 = rect_start
        x2, y2 = current_mouse_pos
        cv2.rectangle(overlay, (min(x1,x2), min(y1,y2)), 
                     (max(x1,x2), max(y1,y2)), (0, 255, 0), 3)
    
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
    
    print("OCR:", repr(text[:200]))
    cv2.imshow('OCR Preview', thresh)
    cv2.waitKey(0)
    cv2.destroyWindow('OCR Preview')
    return text, thresh

cv2.namedWindow('Resizable Cube OCR Tool v4', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Resizable Cube OCR Tool v4', mouse_callback)
cv2.resizeWindow('Resizable Cube OCR Tool v4', w_orig, h_orig + PANEL_HEIGHT)

print("CONTROLS:")
print("  RMB+Drag = New cube")
print("  LMB = Select cube")
print("  HOLD LMB on Rotate buttons = Continuous rotation!")
print("  CLICK: Delete | OCR | Reset")
print("  A/D = Rotate (hotkeys)")
print("  ESC = Exit")

while True:
    display_img = draw_overlay(img_current)
    combined_img = draw_combined_image(display_img)
    cv2.imshow('Resizable Cube OCR Tool v4', combined_img)
    
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
        ocr_process()
    elif key == 27:
        break

cv2.destroyAllWindows()
