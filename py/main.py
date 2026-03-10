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

# Operating modes
MODE_GRID_AUTO = 0
MODE_GRID_MANUAL = 1
current_mode = MODE_GRID_MANUAL

cubes = []
grid_points = []
GRID_STEP = 80
drawing_rect = False
rect_start = (0, 0)
current_mouse_pos = (0, 0)
selected_cube_idx = -1
selected_point = None
selection_rect = None
grid_rotations = {}

# Top control panel (buttons)
PANEL_TOP_HEIGHT = 80
BUTTON_HEIGHT = 60
PANEL_RIGHT_WIDTH = 150

buttons = {
    'rotate_left': {'x': 10, 'y': 10, 'w': 80, 'h': BUTTON_HEIGHT, 'text': '<-10'},
    'rotate_right': {'x': 100, 'y': 10, 'w': 80, 'h': BUTTON_HEIGHT, 'text': '10->'},
    'delete': {'x': 190, 'y': 10, 'w': 80, 'h': BUTTON_HEIGHT, 'text': 'Delete'},
    'ocr': {'x': 280, 'y': 10, 'w': 80, 'h': BUTTON_HEIGHT, 'text': 'OCR'},
    'reset': {'x': 370, 'y': 10, 'w': 80, 'h': BUTTON_HEIGHT, 'text': 'Reset'}
}

NAV_BUTTONS = {
    'grid_auto': {'y': 10, 'h': 60, 'text': 'Auto Grid'},
    'grid_manual': {'y': 80, 'h': 60, 'text': 'Manual Grid'}
}

button_pressed = None
last_rotate_time = 0
ROTATE_INTERVAL = 0.1

def create_grid_points():
    points = []
    for i in range(0, h, GRID_STEP):
        for j in range(0, w, GRID_STEP):
            points.append((j, i))
    return points

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

def rotate_region_around_point(center, angle_deg):
    global img_current, img_gray, grid_rotations
    cx, cy = center
    size = GRID_STEP
    half = size // 2
    x1, y1 = max(0, cx-half), max(0, cy-half)
    x2, y2 = min(w, cx+half), min(h, cy+half)
    if x2 <= x1 or y2 <= y1:
        return
    
    grid_key = (cx, cy)
    if grid_key not in grid_rotations:
        grid_rotations[grid_key] = 0.0
    
    grid_rotations[grid_key] += angle_deg
    total_angle = grid_rotations[grid_key]
    
    region_orig = img_original[y1:y2, x1:x2].copy()
    rows, cols = region_orig.shape[:2]
    M = cv2.getRotationMatrix2D((cols//2, rows//2), total_angle, 1.0)
    rotated = cv2.warpAffine(region_orig, M, (cols, rows), 
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    img_current[y1:y2, x1:x2] = rotated
    img_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
    print(f"Grid[{cx},{cy}] TOTAL: {total_angle:.1f}°")

def global_rotate(angle_deg):
    global img_current, img_gray, angle    
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle + angle_deg, 1.0)
    img_current = cv2.warpAffine(img_original, M, (w, h), 
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    img_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
    angle += angle_deg

def clear_cubes():
    global cubes, selected_cube_idx
    cubes.clear()
    selected_cube_idx = -1
    print("Cubes cleared on mode switch!")

def reset_all():
    global cubes, grid_points, selected_cube_idx, selected_point, img_current, img_gray, angle, button_pressed, grid_rotations, selection_rect
    cubes.clear()
    grid_rotations.clear()
    grid_points = []
    selected_cube_idx = -1
    selected_point = None
    selection_rect = None
    img_current = img_original.copy()
    img_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
    angle = 0
    button_pressed = None
    print("Reset all!")

def draw_right_panel(combined):
    panel_x = w_orig
    panel_y = 0
    
    cv2.rectangle(combined, (panel_x, panel_y), (panel_x + PANEL_RIGHT_WIDTH, h_orig + PANEL_TOP_HEIGHT), (60, 60, 60), -1)
    
    cv2.rectangle(combined, (panel_x + 5, 5), (panel_x + PANEL_RIGHT_WIDTH - 5, 35), (0, 0, 0), -1)
    cv2.putText(combined, "MODES", (panel_x + 10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    for btn_name, btn in NAV_BUTTONS.items():
        btn_y = PANEL_TOP_HEIGHT + btn['y']
        is_active = (btn_name == 'grid_auto' and current_mode == MODE_GRID_AUTO) or \
                   (btn_name == 'grid_manual' and current_mode == MODE_GRID_MANUAL)
        color = (120, 120, 120) if is_active else (80, 80, 80)
        
        cv2.rectangle(combined, (panel_x + 10, btn_y), 
                     (panel_x + PANEL_RIGHT_WIDTH - 10, btn_y + btn['h']), color, -1)
        cv2.rectangle(combined, (panel_x + 10, btn_y), 
                     (panel_x + PANEL_RIGHT_WIDTH - 10, btn_y + btn['h']), (255,255,255), 2)
        
        text_size = cv2.getTextSize(btn['text'], cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        text_x = panel_x + 10 + (PANEL_RIGHT_WIDTH - 20 - text_size[0]) // 2
        text_y = btn_y + (btn['h'] // 2) + (text_size[1] // 2)
        cv2.putText(combined, btn['text'], (int(text_x), int(text_y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
    
    mode_text = "AUTO GRID" if current_mode == MODE_GRID_AUTO else "MANUAL GRID"
    cv2.putText(combined, mode_text, (panel_x + 10, PANEL_TOP_HEIGHT + 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

def mouse_callback(event, x, y, flags, param):
    global drawing_rect, rect_start, current_mouse_pos, selected_cube_idx, button_pressed, last_rotate_time
    global selected_point, selection_rect, current_mode, grid_points
    
    current_time = time.time()
    
    sidebar_x = w_orig
    if event == cv2.EVENT_LBUTTONDOWN and x >= sidebar_x and y >= PANEL_TOP_HEIGHT:
        panel_y = y - PANEL_TOP_HEIGHT
        for btn_name, btn in NAV_BUTTONS.items():
            if (btn['y'] <= panel_y <= btn['y'] + btn['h']):
                old_mode = current_mode
                if btn_name == 'grid_auto':
                    current_mode = MODE_GRID_AUTO
                    grid_points = create_grid_points()
                    print("MODE: AUTO GRID")
                elif btn_name == 'grid_manual':
                    current_mode = MODE_GRID_MANUAL
                    print("MODE: MANUAL GRID")
                
                if old_mode != current_mode:
                    clear_cubes()
                return
    
    if event == cv2.EVENT_LBUTTONDOWN and y >= h_orig:
        panel_y = y - h_orig
        clicked_button = None
        for btn_name, btn in buttons.items():
            if (btn['x'] <= x <= btn['x'] + btn['w'] and 
                btn['y'] <= panel_y <= btn['y'] + btn['h']):
                clicked_button = btn_name
                break
        
        if clicked_button:
            print(f"Clicked: {clicked_button}")
            button_pressed = clicked_button if clicked_button in ['rotate_left', 'rotate_right'] else None
            
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
                last_rotate_time = current_time
                angle_step = -10 if clicked_button == 'rotate_left' else 10
                if selected_cube_idx >= 0:
                    rotate_cube(selected_cube_idx, angle_step)
                elif selected_point:
                    rotate_region_around_point(selected_point, angle_step)
                else:
                    global_rotate(angle_step)
            return
    
    elif event == cv2.EVENT_LBUTTONUP:
        button_pressed = None
        return
    
    elif event == cv2.EVENT_MOUSEMOVE and button_pressed and (flags & cv2.EVENT_FLAG_LBUTTON):
        if button_pressed in ['rotate_left', 'rotate_right'] and current_time - last_rotate_time >= ROTATE_INTERVAL:
            angle_step = -10 if button_pressed == 'rotate_left' else 10
            if selected_cube_idx >= 0:
                rotate_cube(selected_cube_idx, angle_step)
            elif selected_point:
                rotate_region_around_point(selected_point, angle_step)
            else:
                global_rotate(angle_step)
            last_rotate_time = current_time
        return
    
    if x < w_orig and y < h_orig:
        current_mouse_pos = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_cube_idx = -1
            selected_point = None
            
            if current_mode == MODE_GRID_MANUAL:
                for i, cube in enumerate(cubes):
                    cx, cy, cw, ch, _ = cube
                    if (cx-10 <= x <= cx+cw+10) and (cy-10 <= y <= cy+ch+10):
                        selected_cube_idx = i
                        break
            
            if selected_cube_idx == -1 and current_mode == MODE_GRID_AUTO:
                min_dist = float('inf')
                for point in grid_points:
                    dist = ((point[0]-x)**2 + (point[1]-y)**2)**0.5
                    if dist < 25 and dist < min_dist:
                        min_dist = dist
                        selected_point = point
                if selected_point:
                    half = GRID_STEP // 2
                    selection_rect = (selected_point[0]-half, selected_point[1]-half, GRID_STEP, GRID_STEP)
        
        if current_mode == MODE_GRID_MANUAL:
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
    combined_width = w_orig + PANEL_RIGHT_WIDTH
    combined = np.zeros((h_orig + PANEL_TOP_HEIGHT, combined_width, 3), dtype=np.uint8)
    
    combined[:h_orig, :w_orig, :] = img
    
    status_parts = [f"Mode: {'AUTO' if current_mode == MODE_GRID_AUTO else 'MANUAL'}"]
    status_parts.append(f"Cubes: {len(cubes)}")
    if selected_cube_idx >= 0:
        cube = cubes[selected_cube_idx]
        status_parts.append(f"#{selected_cube_idx}: {cube[2]}x{cube[3]}")
    elif selected_point:
        status_parts.append(f"Point: {selected_point}")
    status_parts.append(f"Angle: {angle:.1f}°")
    status = " | ".join(status_parts)
    
    cv2.rectangle(combined, (5, 5), (w_orig-10, 35), (0, 0, 0), -1)
    cv2.putText(combined, status, (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    panel_y = h_orig
    cv2.rectangle(combined, (0, panel_y), (w_orig, h_orig + PANEL_TOP_HEIGHT), (70, 70, 70), -1)
    
    for btn_name, btn in buttons.items():
        btn_y = panel_y + btn['y']
        color = (100, 100, 100) if button_pressed == btn_name else (80, 80, 80)
        
        cv2.rectangle(combined, (btn['x'], btn_y), 
                     (btn['x']+btn['w'], btn_y+btn['h']), color, -1)
        cv2.rectangle(combined, (btn['x'], btn_y), 
                     (btn['x']+btn['w'], btn_y+btn['h']), (255,255,255), 2)
        
        text_size = cv2.getTextSize(btn['text'], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = btn['x'] + (btn['w'] - text_size[0]) // 2
        text_y = btn_y + (btn['h'] // 2) + (text_size[1] // 2)
        cv2.putText(combined, btn['text'], (int(text_x), int(text_y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    draw_right_panel(combined)
    return combined

def draw_overlay(img):
    overlay = img.copy()
    
    if current_mode == MODE_GRID_MANUAL:
        for i, cube in enumerate(cubes):
            x, y, rw, rh, rot = cube
            color = (0, 0, 255) if i == selected_cube_idx else (255, 0, 0)
            thickness = 2 if i == selected_cube_idx else 1
            
            cv2.rectangle(overlay, (x, y), (x+rw, y+rh), color, thickness)
            cv2.putText(overlay, f"#{i}:{rot:.0f}°", (x+5, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    if current_mode == MODE_GRID_AUTO:
        if drawing_rect and rect_start[0] > 0:
            x1, y1 = rect_start
            x2, y2 = current_mouse_pos
            cv2.rectangle(overlay, (min(x1,x2), min(y1,y2)), 
                         (max(x1,x2), max(y1,y2)), (0, 255, 0), 2)
        
        for i in range(0, h, GRID_STEP):
            cv2.line(overlay, (0, i), (w, i), (0, 255, 0), 1)
        for j in range(0, w, GRID_STEP):
            cv2.line(overlay, (j, 0), (j, h), (0, 255, 0), 1)
        
        for point in grid_points:
            color = (0, 255, 255) if point == selected_point else (0, 255, 0)
            cv2.circle(overlay, point, 10, color, -1)
        
        if selection_rect:
            x, y, rw, rh = selection_rect
            cv2.rectangle(overlay, (x, y), (x+rw, y+rh), (0, 0, 255), 2)
    
    if current_mode == MODE_GRID_MANUAL and drawing_rect and rect_start[0] > 0:
        x1, y1 = rect_start
        x2, y2 = current_mouse_pos
        cv2.rectangle(overlay, (min(x1,x2), min(y1,y2)), 
                     (max(x1,x2), max(y1,y2)), (0, 255, 0), 2)
        cv2.circle(overlay, (min(x1,x2), min(y1,y2)), 6, (0, 255, 0), 1)
        cv2.circle(overlay, (max(x1,x2), min(y1,y2)), 6, (0, 255, 0), 1)
        cv2.circle(overlay, (min(x1,x2), max(y1,y2)), 6, (0, 255, 0), 1)
        cv2.circle(overlay, (max(x1,x2), max(y1,y2)), 6, (0, 255, 0), 1)
    
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

grid_points = create_grid_points()

cv2.namedWindow('Cube OCR Tool v1.6', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Cube OCR Tool v1.6', mouse_callback)
cv2.resizeWindow('Cube OCR Tool v1.6', w_orig + PANEL_RIGHT_WIDTH, h_orig + PANEL_TOP_HEIGHT)

print("CONTROLS:")
print("  Auto Grid: LMB=select point, A/D=rotate (FULL 360°!)")
print("  Manual: RMB+Drag = cube, LMB = select cube")
print("  Top buttons: HOLD LMB=continuous rotation")
print("  ESC = Exit")

while True:
    if current_mode == MODE_GRID_AUTO and not grid_points:
        grid_points = create_grid_points()
    
    display_img = draw_overlay(img_current)
    combined_img = draw_combined_image(display_img)
    cv2.imshow('Cube OCR Tool v1.6', combined_img)
    
    key = cv2.waitKey(20) & 0xFF
    
    if selected_cube_idx >= 0 and current_mode == MODE_GRID_MANUAL:
        if key == ord('a') or key == ord('A'):
            rotate_cube(selected_cube_idx, -10)
        elif key == ord('d') or key == ord('D'):
            rotate_cube(selected_cube_idx, 10)
        elif key == 8 or key == 46:
            del cubes[selected_cube_idx]
            selected_cube_idx = -1
    elif selected_point and current_mode == MODE_GRID_AUTO:
        if key == ord('a') or key == ord('A'):
            rotate_region_around_point(selected_point, -10)
        elif key == ord('d') or key == ord('D'):
            rotate_region_around_point(selected_point, 10)
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
