#pylint: disable=no-member
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

# Переменные для растягиваемого куба
drawing_rect = False
rect_start = (0, 0)
selected_rect = None  # Готовый куб для поворота (x, y, width, height)
current_mouse_pos = (0, 0)

def rotate_region_around_rect(rect, angle_deg):
    """Поворот ТОЧНО выделенной области без искажений"""
    global img_current, img_gray
    
    x, y, rw, rh = rect
    
    # Берем ТОЛЬКО выделенную область
    region = img_current[y:y+rh, x:x+rw].copy()
    
    # Поворот относительно ЦЕНТРА выделенной области
    center = (rw // 2, rh // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    
    # Выходной размер = размер входной области
    rotated = cv2.warpAffine(region, M, (rw, rh), 
                            flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_REPLICATE)
    
    # Записываем ТОЛЬКО обратно в выделенную область
    img_current[y:y+rh, x:x+rw] = rotated
    img_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
    print(f"🎯 КУБ ПОВЕРНУТ ({x},{y}) {rw}x{rh} на {angle_deg}°")

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
    print(f"🌍 ГЛОБАЛЬНЫЙ ПОВОРОТ {angle:.1f}°")

def mouse_callback(event, x, y, flags, param):
    global drawing_rect, rect_start, selected_rect, current_mouse_pos
    
    current_mouse_pos = (x, y)
    
    if event == cv2.EVENT_RBUTTONDOWN:
        drawing_rect = True
        rect_start = (x, y)
        selected_rect = None
        print(f"🖱️ Начало выделения ({x}, {y})")
    
    elif event == cv2.EVENT_MOUSEMOVE:
        pass  # current_mouse_pos уже обновлен
    
    elif event == cv2.EVENT_RBUTTONUP:
        if drawing_rect:
            drawing_rect = False
            x1, y1 = rect_start
            x2, y2 = x, y
            rw = abs(x2 - x1)
            rh = abs(y2 - y1)
            
            if rw > 15 and rh > 15:
                selected_rect = (min(x1,x2), min(y1,y2), rw, rh)
                print(f"✅ КУБ СОЗДАН: {selected_rect}")
            else:
                print("❌ Куб слишком мал (мин 15x15)")
    
    elif event == cv2.EVENT_LBUTTONDOWN:
        selected_rect = None
        print("🔄 Выделение снято")

def draw_overlay(img):
    overlay = img.copy()
    
    # Растягиваемый прямоугольник (зеленый)
    if drawing_rect and rect_start[0] > 0:
        x1, y1 = rect_start
        x2, y2 = current_mouse_pos
        cv2.rectangle(overlay, (min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2)), (0, 255, 0), 3)
    
    # Активный куб (красный)
    if selected_rect:
        x, y, rw, rh = selected_rect
        cv2.rectangle(overlay, (x, y), (x+rw, y+rh), (0, 0, 255), 3)
        cv2.putText(overlay, "АКТИВЕН", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
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

cv2.namedWindow('Resizable Cube OCR Tool', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Resizable Cube OCR Tool', mouse_callback)

print("🎮 УПРАВЛЕНИЕ:")
print("  🖱️ ПКМ + ТЯНИ = создать куб любого размера")
print("  🔴 A/D = повернуть АКТИВНЫЙ куб")
print("  🖱️ ЛКМ = снять выделение | S/ENTER = OCR | ESC = выход")

while True:
    display = draw_overlay(img_current)
    
    if drawing_rect:
        status = "🖱️ ТЯНИ ПКМ"
        color = (0, 255, 0)
    elif selected_rect:
        x, y, rw, rh = selected_rect
        status = f"🔴 АКТИВЕН {rw}x{rh}"
        color = (0, 0, 255)
    else:
        status = "🖱️ ПКМ + ТЯНИ"
        color = (0, 255, 255)
    
    cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(display, f"Угол: {angle:.1f}° | Мышь: {current_mouse_pos[0]},{current_mouse_pos[1]}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Resizable Cube OCR Tool', display)
    
    key = cv2.waitKey(20) & 0xFF
    
    if selected_rect and (key == ord('a') or key == ord('A')):
        rotate_region_around_rect(selected_rect, -10)
    elif selected_rect and (key == ord('d') or key == ord('D')):
        rotate_region_around_rect(selected_rect, 10)
    elif not selected_rect and (key == ord('a') or key == ord('A')):
        global_rotate(-10)
    elif not selected_rect and (key == ord('d') or key == ord('D')):
        global_rotate(10)
    
    elif key == 13:  # ENTER
        print("💾 Сохранение final.jpg...")
        cv2.imwrite('final.jpg', img_current)
        text, thresh = ocr_process()
        print("📄 OCR:", repr(text[:200]))
        if 'OCR Preview' in cv2.getWindowProperty('OCR Preview', 0) >= 0:
            cv2.destroyWindow('OCR Preview')
        cv2.imshow('OCR Preview', thresh)
        cv2.waitKey(0)
    
    elif key == ord('s') or key == ord('S'):
        text, thresh = ocr_process()
        print("📄 Быстрый OCR:", repr(text[:200]))
        cv2.imshow('OCR Preview', thresh)
        cv2.waitKey(0)
    
    elif key == 27:  # ESC
        break

cv2.destroyAllWindows()
print("👋 Готово!")
