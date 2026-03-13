import cv2
import numpy as np
from PIL import Image
import pytesseract
import sys
import io
import time
import math

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

pytesseract.pytesseract.tesseract_cmd = r'D:\\tesseract\\tesseract.exe'
FILE_NAME = r'C:\\Users\\user\\Desktop\\pydiplom\\OCRWithGrid\\englishPhoto.jpg'

img_original = cv2.imread(FILE_NAME)
if img_original is None:
    print(f"Error: failed to load {FILE_NAME}")
    sys.exit(1)

h_orig, w_orig = img_original.shape[:2]

# ✅ Изначально показываем ОРИГИНАЛЬНОЕ изображение
img_current = img_original.copy()
img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

# ✅ Белый фон только для деформаций
white_bg = np.full((h_orig, w_orig, 3), 255, dtype=np.uint8)
h, w = h_orig, w_orig
angle = 0

# Operating modes
MODE_GRID_AUTO = 0
MODE_GRID_MANUAL = 1
MODE_PERSPECTIVE = 2
MODE_RESIZE = 3
current_mode = MODE_GRID_AUTO  # ✅ Начинаем с Auto Grid

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

# Perspective variables
perspective_corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)
perspective_selected_corner = -1
perspective_dragging_corner = -1
perspective_drag_offset = np.array([0.0, 0.0])

# ✅ GEOMETRIC MODE - СОВЕРШЕННО НОВЫЕ ПЕРЕМЕННЫЕ
resize_corners = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype=np.float32)
resize_target_corners = resize_corners.copy()
resize_selected_corner = -1
resize_dragging_corner = -1
resize_drag_offset = np.array([0.0, 0.0])

# ✅ Параметры плавности
SMOOTHING_SPEED = 0.7
SMOOTHING_SLOW = 0.3
MIN_POINT_DIST = 60

PANEL_BOTTOM_HEIGHT = 70
PANEL_RIGHT_WIDTH = 150
BUTTON_HEIGHT = 50

buttons = {
    'rotate_left': {'x': 10, 'y': 10, 'w': 70, 'h': BUTTON_HEIGHT, 'text': '<-10'},
    'rotate_right': {'x': 85, 'y': 10, 'w': 70, 'h': BUTTON_HEIGHT, 'text': '10->'},
    'delete': {'x': 160, 'y': 10, 'w': 70, 'h': BUTTON_HEIGHT, 'text': 'Delete'},
    'ocr': {'x': 235, 'y': 10, 'w': 70, 'h': BUTTON_HEIGHT, 'text': 'OCR'},
    'reset': {'x': 310, 'y': 10, 'w': 70, 'h': BUTTON_HEIGHT, 'text': 'Reset'}
}

NAV_BUTTONS = {
    'grid_auto': {'y': 200, 'h': BUTTON_HEIGHT, 'text': 'Auto Grid'},
    'grid_manual': {'y': 260, 'h': BUTTON_HEIGHT, 'text': 'Manual Grid'},
    'perspective': {'y': 320, 'h': BUTTON_HEIGHT, 'text': 'Perspective'},
    'resize': {'y': 380, 'h': BUTTON_HEIGHT, 'text': 'GEOMETRIC'}
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

def apply_perspective_transform():
    global img_current, img_gray
    src_pts = perspective_corners.copy()
    dst_pts = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])
    
    if len(np.unique(src_pts.reshape(-1), axis=0)) < 4:
        return
        
    try:
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        img_current = cv2.warpPerspective(img_original, M, (w, h), 
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
        img_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
    except:
        print("Perspective transform failed")

# ✅ ✅ ✅ ЖЕСТКАЯ ЗАЩИТА ОТ КОЛЛИЗИЙ
def validate_and_constrain_corners(target_corners):
    corners = target_corners.copy()
    
    corners[:, 0] = np.clip(corners[:, 0], MIN_POINT_DIST//2, w_orig - MIN_POINT_DIST//2)
    corners[:, 1] = np.clip(corners[:, 1], MIN_POINT_DIST//2, h_orig - MIN_POINT_DIST//2)
    
    for i in range(4):
        for j in range(i+1, 4):
            while True:
                dist = np.hypot(corners[i][0] - corners[j][0], corners[i][1] - corners[j][1])
                if dist >= MIN_POINT_DIST:
                    break
                
                dx = corners[j][0] - corners[i][0]
                dy = corners[j][1] - corners[i][1]
                norm = np.hypot(dx, dy)
                if norm > 0:
                    dx, dy = dx/norm, dy/norm
                    corners[i][0] -= dx * (MIN_POINT_DIST - dist) * 0.6
                    corners[i][1] -= dy * (MIN_POINT_DIST - dist) * 0.6
                    corners[j][0] += dx * (MIN_POINT_DIST - dist) * 0.4
                    corners[j][1] += dy * (MIN_POINT_DIST - dist) * 0.4
    
    tl, tr, br, bl = corners
    diag13 = np.hypot(tl[0]-br[0], tl[1]-br[1])
    diag24 = np.hypot(tr[0]-bl[0], tr[1]-bl[1])
    
    if diag13 < MIN_POINT_DIST * 1.5 or diag24 < MIN_POINT_DIST * 1.5:
        return False
    
    return corners

def update_smooth_corners():
    global resize_corners, resize_target_corners
    
    valid_target = validate_and_constrain_corners(resize_target_corners)
    
    if valid_target is not False:
        alpha = SMOOTHING_SPEED if resize_dragging_corner != -1 else SMOOTHING_SLOW
        resize_corners[:] = (1 - alpha) * resize_corners + alpha * valid_target

def apply_geometric_transform():
    global img_current, img_gray, resize_corners
    
    src_corners = np.array([[0,0], [w_orig-1,0], [w_orig-1,h_orig-1], [0,h_orig-1]], dtype=np.float32)
    dst_corners = resize_corners.copy()
    
    if len(np.unique(dst_corners.reshape(-1), axis=0)) < 4:
        img_current = img_original.copy()
        img_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
        return
    
    try:
        M = cv2.getPerspectiveTransform(src_corners, dst_corners)
        deformed = cv2.warpPerspective(img_original, M, (w_orig, h_orig), 
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
        
        mask = np.any(deformed < 250, axis=2)
        canvas = white_bg.copy()
        canvas[mask] = deformed[mask]
        
        img_current = canvas
        img_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
        
    except:
        img_current = img_original.copy()
        img_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)

def clear_cubes():
    global cubes, selected_cube_idx
    cubes.clear()
    selected_cube_idx = -1
    print("Cubes cleared!")

def reset_all():
    global cubes, grid_points, selected_cube_idx, selected_point, img_current, img_gray, angle, button_pressed, grid_rotations, selection_rect
    global perspective_corners, perspective_selected_corner, perspective_dragging_corner, perspective_drag_offset
    global resize_corners, resize_target_corners, resize_selected_corner, resize_dragging_corner, resize_drag_offset, w, h
    
    cubes.clear()
    grid_rotations.clear()
    grid_points = []
    selected_cube_idx = -1
    selected_point = None
    selection_rect = None
    
    perspective_corners = np.array([[0,0],[w_orig,0],[w_orig,h_orig],[0,h_orig]], dtype=np.float32)
    perspective_selected_corner = -1
    perspective_dragging_corner = -1
    perspective_drag_offset = np.array([0.0, 0.0])
    
    resize_corners = np.array([[0,0], [w_orig-1,0], [w_orig-1,h_orig-1], [0,h_orig-1]], dtype=np.float32)
    resize_target_corners = resize_corners.copy()
    resize_selected_corner = -1
    resize_dragging_corner = -1
    resize_drag_offset = np.array([0.0, 0.0])
    
    img_current = img_original.copy()
    img_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
    w, h = w_orig, h_orig
    angle = 0
    button_pressed = None
    print("✅ Reset - ORIGINAL IMAGE!")

def switch_to_geometry_mode():
    """✅ Специальная инициализация GEOMETRY с белым фоном"""
    global img_current, img_gray, resize_corners, resize_target_corners
    
    # Немного сдвигаем углы чтобы сразу показать белый фон
    offset = 30
    resize_corners = np.array([
        [offset, offset], 
        [w_orig-1-offset, offset], 
        [w_orig-1-offset, h_orig-1-offset], 
        [offset, h_orig-1-offset]
    ], dtype=np.float32)
    resize_target_corners = resize_corners.copy()
    
    # Применяем трансформацию сразу
    apply_geometric_transform()
    print("🚀 GEOMETRY mode - WHITE BACKGROUND activated!")

def draw_right_panel(combined):
    panel_x = w_orig
    
    cv2.rectangle(combined, (panel_x, 0), (panel_x + PANEL_RIGHT_WIDTH, h_orig + PANEL_BOTTOM_HEIGHT), (60, 60, 60), -1)
    
    info_y = 10
    cv2.rectangle(combined, (panel_x + 5, info_y - 5), (panel_x + PANEL_RIGHT_WIDTH - 5, info_y + 200), (40, 40, 40), -1)
    cv2.rectangle(combined, (panel_x + 5, info_y - 5), (panel_x + PANEL_RIGHT_WIDTH - 5, info_y + 200), (100, 100, 100), 1)
    
    cv2.putText(combined, "INFO", (panel_x + 10, info_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    info_y += 28
    mode_text = "AUTO" if current_mode == MODE_GRID_AUTO else "MANUAL" if current_mode == MODE_GRID_MANUAL else "PERSPECTIVE" if current_mode == MODE_PERSPECTIVE else "GEOMETRIC"
    cv2.putText(combined, f"Mode: {mode_text}", (panel_x + 10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    info_y += 22
    
    if current_mode == MODE_RESIZE:
        cv2.putText(combined, f"α={SMOOTHING_SPEED:.1f}/{SMOOTHING_SLOW:.1f}", (panel_x + 10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
        info_y += 20
        
        tl, tr, br, bl = resize_corners
        diag13 = np.hypot(tl[0]-br[0], tl[1]-br[1])
        cv2.putText(combined, f"Diag: {diag13:.0f}px", (panel_x + 10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 0), 1)

    cv2.rectangle(combined, (panel_x + 5, 220), (panel_x + PANEL_RIGHT_WIDTH - 5, 250), (0, 0, 0), -1)
    cv2.putText(combined, "MODES", (panel_x + 10, 237), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    for btn_name, btn in NAV_BUTTONS.items():
        btn_y = btn['y']
        is_active = (btn_name == 'grid_auto' and current_mode == MODE_GRID_AUTO) or \
                   (btn_name == 'grid_manual' and current_mode == MODE_GRID_MANUAL) or \
                   (btn_name == 'perspective' and current_mode == MODE_PERSPECTIVE) or \
                   (btn_name == 'resize' and current_mode == MODE_RESIZE)
        
        color = (120, 120, 120) if is_active else (80, 80, 80)
        cv2.rectangle(combined, (panel_x + 10, btn_y), (panel_x + PANEL_RIGHT_WIDTH - 10, btn_y + btn['h']), color, -1)
        cv2.rectangle(combined, (panel_x + 10, btn_y), (panel_x + PANEL_RIGHT_WIDTH - 10, btn_y + btn['h']), (255,255,255), 2)
        
        text_size = cv2.getTextSize(btn['text'], cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
        text_x = panel_x + 10 + (PANEL_RIGHT_WIDTH - 20 - text_size[0]) // 2
        text_y = btn_y + (btn['h'] // 2) + (text_size[1] // 2)
        cv2.putText(combined, btn['text'], (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

# ✅ ✅ ✅ ИСПРАВЛЕННЫЙ mouse_callback - ТОЧКИ ДВИГУТСЯ ТОЛЬКО ПО ЛКМ
def mouse_callback(event, x, y, flags, param):
    global drawing_rect, rect_start, current_mouse_pos, selected_cube_idx, button_pressed, last_rotate_time
    global selected_point, selection_rect, current_mode, grid_points
    global perspective_corners, perspective_selected_corner, perspective_dragging_corner, perspective_drag_offset
    global resize_corners, resize_target_corners, resize_selected_corner, resize_dragging_corner, resize_drag_offset
    
    current_time = time.time()
    
    sidebar_x = w_orig
    if event == cv2.EVENT_LBUTTONDOWN and x >= sidebar_x:
        panel_y = y
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
                elif btn_name == 'perspective':
                    current_mode = MODE_PERSPECTIVE
                    print("MODE: PERSPECTIVE")
                elif btn_name == 'resize':
                    current_mode = MODE_RESIZE
                    switch_to_geometry_mode()  # ✅ Специальная инициализация с белым фоном
                    print("🚀 MODE: PERFECT SMOOTH GEOMETRY v4.1 - CLICK ONLY!")
                    
                if old_mode != current_mode:
                    clear_cubes()
                return
    
    if event == cv2.EVENT_LBUTTONDOWN and y >= h_orig:
        panel_y = y - h_orig
        clicked_button = None
        for btn_name, btn in buttons.items():
            if (btn['x'] <= x <= btn['x'] + btn['w'] and btn['y'] <= panel_y <= btn['y'] + btn['h']):
                clicked_button = btn_name
                break
        
        if clicked_button:
            print(f"Clicked: {clicked_button}")
            button_pressed = clicked_button if clicked_button in ['rotate_left', 'rotate_right'] else None
            
            if clicked_button == 'delete' and selected_cube_idx >= 0:
                del cubes[selected_cube_idx]
                selected_cube_idx = -1
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
        if perspective_dragging_corner != -1:
            perspective_dragging_corner = -1
        if resize_dragging_corner != -1:
            resize_dragging_corner = -1
            print("📄 Corner released")
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
        
        # ✅ ✅ ✅ GEOMETRY MODE v4.1 - ТОЛЬКО ЛКМ!
        if current_mode == MODE_RESIZE:
            corner_radius = 22
            
            # ✅ 1. КЛИК - мгновенный захват
            if event == cv2.EVENT_LBUTTONDOWN:
                for i, corner in enumerate(resize_corners):
                    if math.hypot(corner[0]-x, corner[1]-y) < corner_radius:
                        resize_drag_offset = corner - np.array([x, y])
                        resize_dragging_corner = i
                        resize_selected_corner = i
                        resize_target_corners[i] = np.array([x, y])  # ✅ МГНОВЕННО!
                        print(f"📄 Corner {i} CAPTURED - smooth tracking STARTED!")
                        return
            
            # ✅ 2. ДРАГ - только при зажатой ЛКМ
            elif resize_dragging_corner != -1 and (flags & cv2.EVENT_FLAG_LBUTTON):
                current_pos = np.array([x, y])
                target_pos = current_pos + resize_drag_offset
                resize_target_corners[resize_dragging_corner] = target_pos
                return
            
            # ✅ 3. Наведение НЕ ДВИГАЕТ точки!
        
        # Остальные режимы
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_cube_idx = -1
            selected_point = None
            
            if current_mode == MODE_PERSPECTIVE:
                for i, corner in enumerate(perspective_corners):
                    if math.hypot(corner[0]-x, corner[1]-y) < 20:
                        perspective_drag_offset = corner - np.array([x, y])
                        perspective_dragging_corner = i
                        perspective_selected_corner = i
                        return
            
            if current_mode == MODE_GRID_MANUAL:
                for i, cube in enumerate(cubes):
                    cx, cy, cw, ch, _ = cube
                    if (cx-10 <= x <= cx+cw+10) and (cy-10 <= y <= cy+ch+10):
                        selected_cube_idx = i
                        break
            
            elif current_mode == MODE_GRID_AUTO:
                min_dist = float('inf')
                for point in grid_points:
                    dist = ((point[0]-x)**2 + (point[1]-y)**2)**0.5
                    if dist < 25 and dist < min_dist:
                        min_dist = dist
                        selected_point = point
                if selected_point:
                    half = GRID_STEP // 2
                    selection_rect = (selected_point[0]-half, selected_point[1]-half, GRID_STEP, GRID_STEP)
        
        elif current_mode == MODE_PERSPECTIVE and perspective_dragging_corner != -1 and (flags & cv2.EVENT_FLAG_LBUTTON):
            current_pos = np.array([x, y])
            perspective_corners[perspective_dragging_corner] = current_pos + perspective_drag_offset
            apply_perspective_transform()
            return
        
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
    combined_height = h_orig + PANEL_BOTTOM_HEIGHT
    
    combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    combined[:h_orig, :w_orig, :] = img
    
    panel_y = h_orig
    cv2.rectangle(combined, (0, panel_y), (w_orig, panel_y + PANEL_BOTTOM_HEIGHT), (70, 70, 70), -1)
    
    for btn_name, btn in buttons.items():
        btn_y = panel_y + btn['y']
        color = (100, 100, 100) if button_pressed == btn_name else (80, 80, 80)
        cv2.rectangle(combined, (btn['x'], btn_y), (btn['x']+btn['w'], btn_y+btn['h']), color, -1)
        cv2.rectangle(combined, (btn['x'], btn_y), (btn['x']+btn['w'], btn_y+btn['h']), (255,255,255), 2)
        
        text_size = cv2.getTextSize(btn['text'], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = btn['x'] + (btn['w'] - text_size[0]) // 2
        text_y = btn_y + (btn['h'] // 2) + (text_size[1] // 2)
        cv2.putText(combined, btn['text'], (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
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
            cv2.putText(overlay, f"#{i}:{rot:.0f}°", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    if current_mode == MODE_GRID_AUTO:
        if drawing_rect and rect_start[0] > 0:
            x1, y1 = rect_start
            x2, y2 = current_mouse_pos
            cv2.rectangle(overlay, (min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2)), (0, 255, 0), 2)
        
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
    
    if current_mode == MODE_PERSPECTIVE:
        for i, corner in enumerate(perspective_corners):
            corner_int = tuple(corner.astype(int))
            if i == perspective_dragging_corner:
                color, size = (0, 255, 255), 22
                cv2.circle(overlay, corner_int, size+8, color, 3)
            elif i == perspective_selected_corner:
                color, size = (255, 255, 255), 20
            else:
                color, size = (0, 255, 0), 18
            cv2.circle(overlay, corner_int, size, color, -1)
            cv2.circle(overlay, corner_int, size, (0, 0, 0), 3)
            cv2.putText(overlay, str(i), corner_int, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        
        pts = perspective_corners.astype(int)
        cv2.line(overlay, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), 4)
        cv2.line(overlay, tuple(pts[1]), tuple(pts[2]), (0, 0, 255), 4)
        cv2.line(overlay, tuple(pts[2]), tuple(pts[3]), (0, 0, 255), 4)
        cv2.line(overlay, tuple(pts[3]), tuple(pts[0]), (0, 0, 255), 4)
    
    # ✅ GEOMETRY MODE v4.1
    if current_mode == MODE_RESIZE:
        for i, corner in enumerate(resize_corners):
            corner_int = tuple(corner.astype(int))
            
            if i == resize_dragging_corner:
                color, size = (0, 255, 255), 16
                cv2.circle(overlay, corner_int, size+10, color, 4)
                cv2.circle(overlay, corner_int, size+6, (255,255,255), 2)
            elif i == resize_selected_corner:
                color, size = (255, 255, 0), 14
                cv2.circle(overlay, corner_int, size+8, (0,0,0), 2)
            else:
                color, size = (0, 255, 0), 12
            
            cv2.circle(overlay, corner_int, size, color, -1)
            cv2.circle(overlay, corner_int, size+3, (0,0,0), 2)
            cv2.putText(overlay, str(i), (corner_int[0]-10, corner_int[1]-12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        
        pts = resize_corners.astype(int)
        cv2.line(overlay, tuple(pts[0]), tuple(pts[1]), (255, 255, 255), 5)
        cv2.line(overlay, tuple(pts[1]), tuple(pts[2]), (255, 255, 255), 5)
        cv2.line(overlay, tuple(pts[2]), tuple(pts[3]), (255, 255, 255), 5)
        cv2.line(overlay, tuple(pts[3]), tuple(pts[0]), (255, 255, 255), 5)
    
    if current_mode == MODE_GRID_MANUAL and drawing_rect and rect_start[0] > 0:
        x1, y1 = rect_start
        x2, y2 = current_mouse_pos
        cv2.rectangle(overlay, (min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2)), (0, 255, 0), 2)
    
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

cv2.namedWindow('Cube OCR Tool v4.1 - CLICK ONLY GEOMETRY', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Cube OCR Tool v4.1 - CLICK ONLY GEOMETRY', mouse_callback)
cv2.resizeWindow('Cube OCR Tool v4.1 - CLICK ONLY GEOMETRY', w_orig + PANEL_RIGHT_WIDTH, h_orig + PANEL_BOTTOM_HEIGHT)

while True:
    if current_mode == MODE_GRID_AUTO and not grid_points:
        grid_points = create_grid_points()
    
    # ✅ ✅ ✅ ГЛАВНЫЙ ЦИКЛ - ПЛАВНОСТЬ ТОЛЬКО при активном dragging
    if current_mode == MODE_RESIZE and resize_dragging_corner != -1:
        update_smooth_corners()
        apply_geometric_transform()
    
    display_img = draw_overlay(img_current)
    combined_img = draw_combined_image(display_img)
    cv2.imshow('Cube OCR Tool v4.1 - CLICK ONLY GEOMETRY', combined_img)
    
    key = cv2.waitKey(1) & 0xFF
    
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
