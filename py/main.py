# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import pytesseract
import time

pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'

class OCRCorrector:
    def __init__(self, root):
        self.root = root
        self.root.title("Устранение искажений сканированного текста (OCR Grid v3.0)")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1e1e1e')
        
        # Изображения
        self.original_img = None
        self.processed_img = None
        self.left_photo = None
        self.right_photo = None
        
        # Переменные для режимов
        self.points = []
        self.base_points = []
        self.triangles = []
        self.dragging_idx = None
        self.dragging_group = []
        self.last_update_time = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        self.current_mode = 0  # 0=Grid, 1=Perspective, 2=Geometry
        self.grid_size_var = tk.IntVar(value=4)
        self.grid_mode_var = tk.StringVar(value="Треугольники")
        
        # Perspective
        self.perspective_corners = None
        
        # Geometry 
        self.resize_corners = None
        
        self.image_offset_x = 0
        self.image_offset_y = 0
        self.current_scale = 1.0
        
        # Сохранение промежуточного результата
        self.current_base_img = None
        
        self.setup_ui()
        self.root.after(500, self.show_welcome_alert)

    def show_welcome_alert(self):
        messagebox.showinfo("🎉 Добро пожаловать!", 
                           "📁 Выберите изображение для начала работы\n\n"
                           "🛠 Режимы работы:\n"
                           "🔳 Треугольная сетка - точная деформация\n"
                           "⬜ Квадратная сетка - оптимизированная\n"
                           "📐 Перспектива 4 точки - исправление углов\n"
                           "🎛 Геометрия + белый фон - кадрирование")

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        top_panel = ttk.Frame(main_frame)
        top_panel.pack(fill=tk.X, pady=(0,10))
        
        ttk.Button(top_panel, text="📁 Загрузить изображение", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_panel, text="🔄 Сброс", command=self.reset_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_panel, text="📝 OCR PRO", command=self.run_ocr).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_panel, text="💾 Сохранить", command=self.save_image).pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(top_panel, orient='vertical').pack(side=tk.LEFT, padx=20, fill=tk.Y)
        
        ttk.Label(top_panel, text="Плотность сетки:").pack(side=tk.LEFT, padx=(0,5))
        self.grid_spin = ttk.Spinbox(top_panel, from_=1, to=7, width=5, textvariable=self.grid_size_var)
        self.grid_spin.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(top_panel, text="Тип сетки:").pack(side=tk.LEFT, padx=(10,5))
        self.mode_combo = ttk.Combobox(top_panel, textvariable=self.grid_mode_var, 
                                     values=["Треугольники", "Квадраты"], width=12, state="readonly")
        self.mode_combo.pack(side=tk.LEFT, padx=5)
        
        mode_frame = ttk.LabelFrame(main_frame, text="Режим работы")
        mode_frame.pack(fill=tk.X, pady=(10,0))
        
        self.mode_var = tk.IntVar(value=0)
        modes = [
            ("🔳 Сетка (Треугольники/Квадраты)", 0),
            ("📐 Перспектива 4 точки", 1), 
            ("🎛 Геометрия + белый фон", 2)
        ]
        
        for i, (text, mode_id) in enumerate(modes):
            btn = ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, 
                                value=mode_id, command=self.on_mode_change)
            btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        img_frame = ttk.Frame(main_frame)
        img_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.LabelFrame(img_frame, text="📄 Исходное (Редактирование)")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))
        self.left_canvas = tk.Canvas(left_frame, bg='#2b2b2b', highlightthickness=0)
        self.left_canvas.pack(fill=tk.BOTH, expand=True)
        
        right_frame = ttk.LabelFrame(img_frame, text="✅ РЕЗУЛЬТАТ")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10,0))
        self.right_canvas = tk.Canvas(right_frame, bg='#2b2b2b', highlightthickness=0)
        self.right_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.left_canvas.bind("<Button-1>", self.on_click)
        self.left_canvas.bind("<B1-Motion>", self.on_drag)
        self.left_canvas.bind("<ButtonRelease-1>", self.on_release)
        self.left_canvas.bind("<Configure>", self.on_canvas_resize)
        self.right_canvas.bind("<Configure>", self.on_canvas_resize)

    def on_mode_change(self):
        mode = self.mode_var.get()
        self.current_mode = mode  # ИСПРАВЛЕНИЕ: Сразу обновляем системный режим
        print(f"🎛 РЕЖИМ ИЗМЕНЕН НА: {mode}")
        
        if self.processed_img is not None and self.original_img is not None:
            self.current_base_img = self.processed_img.copy()
            print("💾 Текущий результат сохранен как базовое изображение")
        
        if mode == 0:
            self.init_grid()
        elif mode == 1:
            self.init_perspective()
        elif mode == 2:
            self.init_geometry()
        
        self.update_displays()

    def on_canvas_resize(self, event):
        self.update_displays()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.processed_img = self.original_img.copy()
                self.current_base_img = None
                print(f"✅ Изображение загружено: {self.original_img.shape}")
                self.reset_all()
                self.update_displays()

    def save_image(self):
        if self.processed_img is None:
            messagebox.showwarning("Внимание", "Нет обработанного изображения для сохранения")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All files", "*.*")],
            title="Сохранить результат как..."
        )
        
        if path:
            try:
                final_bgr = cv2.cvtColor(self.processed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path, final_bgr)
                messagebox.showinfo("Успех", f"Изображение успешно сохранено:\n{path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить изображение: {e}")

    def init_grid(self):
        if self.original_img is None: return
        base_img = self.current_base_img if self.current_base_img is not None else self.original_img
        
        try:
            n = int(self.grid_size_var.get())
        except:
            n = 4
            
        if n > 7: n = 7
        self.grid_size_var.set(n)
            
        h, w = base_img.shape[:2]
        self.points = [[float(x), float(y)] for y in np.linspace(0, h, n + 1) for x in np.linspace(0, w, n + 1)]
        self.base_points = [p[:] for p in self.points]
        
        self.triangles = []
        cols = n + 1
        mode = self.grid_mode_var.get()

        for j in range(n):
            for i in range(n):
                p1, p2, p3, p4 = j*cols+i, j*cols+i+1, (j+1)*cols+i, (j+1)*cols+i+1
                if mode == "Треугольники":
                    self.triangles.append((p1, p2, p3))
                    self.triangles.append((p2, p4, p3))
                else: 
                    self.triangles.append((p1, p2, p4, p3))
        
        self.process_warp(base_img=base_img)

    def init_perspective(self):
        if self.original_img is None: return
        base_img = self.current_base_img if self.current_base_img is not None else self.original_img
        h, w = base_img.shape[:2]
        self.perspective_corners = np.array([
            [50, 50],
            [w-50, 50], 
            [w-50, h-50],
            [50, h-50]
        ], dtype=np.float32)
        self.apply_perspective_transform(base_img=base_img)

    def init_geometry(self):
        if self.original_img is None: return
        base_img = self.current_base_img if self.current_base_img is not None else self.original_img
        h, w = base_img.shape[:2]
        offset = 40
        self.resize_corners = np.array([
            [offset, offset],
            [w-offset-1, offset],
            [w-offset-1, h-offset-1],
            [offset, h-offset-1]
        ], dtype=np.float32)
        self.apply_geometric_transform(base_img=base_img)

    def process_warp(self, base_img, quality=cv2.INTER_LINEAR):
        if not self.points or not self.triangles: 
            self.processed_img = base_img.copy()
            return
        
        h, w = base_img.shape[:2]
        out_img = np.full_like(base_img, 255)
        
        for cell in self.triangles:
            src_pts = np.float32([self.points[i] for i in cell])
            dst_pts = np.float32([self.base_points[i] for i in cell])
            
            x, y, w_b, h_b = cv2.boundingRect(dst_pts)
            dst_rect = dst_pts - (x, y)
            
            mask = np.zeros((h_b, w_b), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(dst_rect), 255)
            
            sx, sy, sw, sh = cv2.boundingRect(src_pts)
            sx_end, sy_end = min(w, sx + sw), min(h, sy + sh)
            img_patch = base_img[sy:sy_end, sx:sx_end]
            if img_patch.size == 0: continue

            src_in_patch = src_pts - (sx, sy)
            
            if len(cell) == 3:
                matrix = cv2.getAffineTransform(np.float32(src_in_patch), np.float32(dst_rect))
                warped_patch = cv2.warpAffine(img_patch, matrix, (w_b, h_b), flags=quality, borderMode=cv2.BORDER_REFLECT_101)
            else:
                matrix = cv2.getPerspectiveTransform(np.float32(src_in_patch), np.float32(dst_rect))
                warped_patch = cv2.warpPerspective(img_patch, matrix, (w_b, h_b), flags=quality, borderMode=cv2.BORDER_REFLECT_101)
            
            slice_y, slice_x = slice(y, min(y + h_b, h)), slice(x, min(x + w_b, w))
            ph, pw = out_img[slice_y, slice_x].shape[:2]
            
            out_img[slice_y, slice_x][mask[:ph, :pw] == 255] = warped_patch[:ph, :pw][mask[:ph, :pw] == 255]
        
        self.processed_img = out_img

    def apply_perspective_transform(self, base_img):
        if base_img is None or self.perspective_corners is None: return
        h, w = base_img.shape[:2]
        src_pts = self.perspective_corners
        dst_pts = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])
        try:
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            self.processed_img = cv2.warpPerspective(base_img, M, (w, h))
        except Exception:
            self.processed_img = base_img.copy()

    def apply_geometric_transform(self, base_img):
        if base_img is None or self.resize_corners is None: return
        h, w = base_img.shape[:2]
        src_corners = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]])
        dst_corners = self.resize_corners
        white_bg = np.full((h, w, 3), 255, dtype=np.uint8)
        try:
            M = cv2.getPerspectiveTransform(src_corners, dst_corners)
            deformed = cv2.warpPerspective(base_img, M, (w, h), 
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_CONSTANT, 
                                        borderValue=(255,255,255))
            
            gray_def = cv2.cvtColor(deformed, cv2.COLOR_RGB2GRAY)
            mask = gray_def < 240
            canvas = white_bg.copy()
            canvas[mask] = deformed[mask]
            self.processed_img = canvas
        except Exception:
            self.processed_img = base_img.copy()

    def get_scale(self):
        if self.original_img is None: return 1.0
        h, w = self.original_img.shape[:2]
        cw, ch = self.left_canvas.winfo_width(), self.left_canvas.winfo_height()
        if cw < 10 or ch < 10: return 1.0
        return min(cw/w, ch/h)

    def on_click(self, event):
        if self.original_img is None: return
        self.dragging_idx = None
        self.dragging_group = []
        mode = self.mode_var.get()
        scale = self.get_scale()
        
        POINT_TOLERANCE = 15 
        LINE_TOLERANCE = 8

        if mode == 0:
            for i, p in enumerate(self.points):
                sx = p[0] * scale + self.image_offset_x
                sy = p[1] * scale + self.image_offset_y
                if abs(sx - event.x) < POINT_TOLERANCE and abs(sy - event.y) < POINT_TOLERANCE:
                    self.dragging_idx = i
                    return

            for cell in self.triangles:
                n = len(cell)
                for i in range(n):
                    idx1, idx2 = cell[i], cell[(i + 1) % n]
                    p1 = np.array([self.points[idx1][0] * scale + self.image_offset_x,
                                   self.points[idx1][1] * scale + self.image_offset_y])
                    p2 = np.array([self.points[idx2][0] * scale + self.image_offset_x,
                                   self.points[idx2][1] * scale + self.image_offset_y])
                    
                    mouse = np.array([event.x, event.y])
                    line_vec = p2 - p1
                    mouse_vec = mouse - p1
                    line_len = np.linalg.norm(line_vec)
                    if line_len == 0: continue
                    
                    t = max(0, min(1, np.dot(mouse_vec, line_vec) / (line_len**2)))
                    projection = p1 + t * line_vec
                    dist = np.linalg.norm(mouse - projection)

                    if dist < LINE_TOLERANCE:
                        self.dragging_group = [idx1, idx2]
                        self.last_mouse_x, self.last_mouse_y = event.x, event.y
                        return

            for cell in self.triangles:
                poly = []
                for idx in cell:
                    poly.append([self.points[idx][0] * scale + self.image_offset_x,
                                 self.points[idx][1] * scale + self.image_offset_y])
                if cv2.pointPolygonTest(np.array(poly, dtype=np.int32), (event.x, event.y), False) >= 0:
                    self.dragging_group = list(cell)
                    self.last_mouse_x, self.last_mouse_y = event.x, event.y
                    return
                    
        elif mode == 1:
            for i in range(4):
                corner = self.perspective_corners[i]
                sx = corner[0] * scale + self.image_offset_x
                sy = corner[1] * scale + self.image_offset_y
                if abs(sx - event.x) < 30 and abs(sy - event.y) < 30:
                    self.dragging_idx = i
                    break
                    
        elif mode == 2:
            for i in range(4):
                corner = self.resize_corners[i]
                sx = corner[0] * scale + self.image_offset_x
                sy = corner[1] * scale + self.image_offset_y
                if abs(sx - event.x) < 30 and abs(sy - event.y) < 30:
                    self.dragging_idx = i
                    break

    def on_drag(self, event):
        if (self.dragging_idx is None and not self.dragging_group) or self.original_img is None: 
            return
        
        curr_time = time.time()
        if curr_time - self.last_update_time < 0.03: return
        self.last_update_time = curr_time

        scale = self.get_scale()
        h, w = self.original_img.shape[:2]
        base_img = self.current_base_img if self.current_base_img is not None else self.original_img
        
        if self.current_mode == 0 and self.dragging_idx is not None:
            nx = max(0, min(w-1, (event.x - self.image_offset_x) / scale))
            ny = max(0, min(h-1, (event.y - self.image_offset_y) / scale))
            self.points[self.dragging_idx] = [nx, ny]
            self.process_warp(base_img, cv2.INTER_NEAREST)
            
        elif self.current_mode == 0 and self.dragging_group:
            dx = (event.x - self.last_mouse_x) / scale
            dy = (event.y - self.last_mouse_y) / scale
            for idx in self.dragging_group:
                new_x = self.points[idx][0] + dx
                new_y = self.points[idx][1] + dy
                self.points[idx] = [max(0, min(w, new_x)), max(0, min(h, new_y))]
            self.last_mouse_x, self.last_mouse_y = event.x, event.y
            self.process_warp(base_img, cv2.INTER_NEAREST)
            
        elif self.current_mode == 1:
            # ИСПРАВЛЕНИЕ: Мы обновляем только координаты и отрисовываем линии, 
            # но НЕ ВЫЗЫВАЕМ тяжелый apply_perspective_transform чтобы не лагало
            nx = max(0, min(w-1, (event.x - self.image_offset_x) / scale))
            ny = max(0, min(h-1, (event.y - self.image_offset_y) / scale))
            self.perspective_corners[self.dragging_idx] = [nx, ny]
            
        elif self.current_mode == 2:
            # ИСПРАВЛЕНИЕ: Аналогично для геометрии
            nx = max(0, min(w-1, (event.x - self.image_offset_x) / scale))
            ny = max(0, min(h-1, (event.y - self.image_offset_y) / scale))
            self.resize_corners[self.dragging_idx] = [nx, ny]
        
        self.update_displays()

    def on_release(self, event):
        if self.dragging_idx is not None or self.dragging_group:
            self.dragging_idx = None
            self.dragging_group = []
            
            mode = self.mode_var.get()
            base_img = self.current_base_img if self.current_base_img is not None else self.original_img
            
            # ИСПРАВЛЕНИЕ: Вызываем тяжелые вычисления только когда мышь отпущена!
            if mode == 0:
                self.process_warp(base_img, cv2.INTER_LINEAR)
            elif mode == 1:
                self.apply_perspective_transform(base_img)
            elif mode == 2:
                self.apply_geometric_transform(base_img)
            
            self.update_displays()

    def update_displays(self):
        if self.original_img is None: return
        
        raw_scale = self.get_scale()
        self.current_scale = raw_scale * 0.95 
        h, w = self.original_img.shape[:2]
        sw, sh = int(w * self.current_scale), int(h * self.current_scale)
        self.image_offset_x = max(0, (self.left_canvas.winfo_width() - sw) // 2)
        self.image_offset_y = max(0, (self.left_canvas.winfo_height() - sh) // 2)

        img_left = cv2.resize(self.original_img, (sw, sh))
        self.left_photo = ImageTk.PhotoImage(Image.fromarray(img_left))
        self.left_canvas.delete("all")
        self.left_canvas.create_image(self.image_offset_x + sw//2, 
                                    self.image_offset_y + sh//2, 
                                    image=self.left_photo)
        self.draw_overlay()

        if self.processed_img is not None:
            rw, rh = self.right_canvas.winfo_width(), self.right_canvas.winfo_height()
            rscale = min(rw/w, rh/h) * 0.9
            img_right = cv2.resize(self.processed_img, (int(w*rscale), int(h*rscale)))
            self.right_photo = ImageTk.PhotoImage(Image.fromarray(img_right))
            self.right_canvas.delete("all")
            self.right_canvas.create_image(rw//2, rh//2, image=self.right_photo)

    def draw_overlay(self):
        mode = self.mode_var.get()
        
        if mode == 0 and self.points:
            for cell in self.triangles:
                pts_coords = []
                for idx in cell:
                    p = np.array(self.points[idx])
                    pts_coords.extend([p[0] * self.current_scale + self.image_offset_x, 
                                       p[1] * self.current_scale + self.image_offset_y])
                self.left_canvas.create_polygon(pts_coords, fill="", outline="#00ff88", width=1)
            
            for i, p in enumerate(self.points):
                sx = p[0] * self.current_scale + self.image_offset_x
                sy = p[1] * self.current_scale + self.image_offset_y
                fill_col = "#ff4444" if i == self.dragging_idx else "#44ff44"
                self.left_canvas.create_oval(sx-6, sy-6, sx+6, sy+6, fill=fill_col, outline="white", width=2)
                
        elif mode == 1 and self.perspective_corners is not None:
            corners = []
            for i, corner in enumerate(self.perspective_corners):
                sx = corner[0] * self.current_scale + self.image_offset_x
                sy = corner[1] * self.current_scale + self.image_offset_y
                corners.append((sx, sy))
                
                size = 15 if self.dragging_idx == i else 10
                fill_col = "#ffff44" if self.dragging_idx == i else "#44ff44"
                self.left_canvas.create_oval(sx-size, sy-size, sx+size, sy+size, 
                                            fill=fill_col, outline="black", width=2)
                self.left_canvas.create_text(sx, sy-20, text=str(i), 
                                            fill="black", font=("Arial", 12, "bold"))
            
            for i in range(4):
                j = (i + 1) % 4
                self.left_canvas.create_line(corners[i][0], corners[i][1], 
                                            corners[j][0], corners[j][1], 
                                            fill="#ffaa00", width=4)
        
        elif mode == 2 and self.resize_corners is not None:
            corners = []
            for i, corner in enumerate(self.resize_corners):
                sx = corner[0] * self.current_scale + self.image_offset_x
                sy = corner[1] * self.current_scale + self.image_offset_y
                corners.append((sx, sy))
                
                size = 18 if self.dragging_idx == i else 12
                fill_col = "#ffaa00" if self.dragging_idx == i else "#aaff00"
                self.left_canvas.create_oval(sx-size, sy-size, sx+size, sy+size, 
                                            fill=fill_col, outline="black", width=3)
                self.left_canvas.create_text(sx, sy-25, text=str(i), 
                                            fill="white", font=("Arial", 14, "bold"))
            
            for i in range(4):
                j = (i + 1) % 4
                self.left_canvas.create_line(corners[i][0], corners[i][1], 
                                            corners[j][0], corners[j][1], 
                                            fill="#ff4400", width=6)

    def reset_all(self):
        self.points = []
        self.base_points = []
        self.triangles = []
        self.perspective_corners = None
        self.resize_corners = None
        self.dragging_idx = None
        self.dragging_group = []
        self.current_base_img = None
        if self.original_img is not None:
            self.processed_img = self.original_img.copy()
        self.update_displays()

    def run_ocr(self):
        if self.processed_img is None:
            messagebox.showwarning("Ошибка", "Сначала загрузите изображение!")
            return
        
        cv2.imwrite('final_corrected.jpg', cv2.cvtColor(self.processed_img, cv2.COLOR_RGB2BGR))
        
        gray = cv2.cvtColor(self.processed_img, cv2.COLOR_RGB2GRAY)
        denoised = cv2.medianBlur(gray, 3)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        cv2.imwrite('ocr_processed.jpg', binary)
        
        config = '--oem 3 --psm 6'
        text = pytesseract.image_to_string(Image.fromarray(binary), lang='eng', config=config)
        text = text.strip()
        
        with open('ocr_result.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        
        messagebox.showinfo("OCR завершено!", 
                           f"Файлы сохранены:\n"
                           f"📁 final_corrected.jpg\n"
                           f"📁 ocr_processed.jpg\n"
                           f"📄 ocr_result.txt\n\n"
                           f"Текст:\n{text[:300]}...")

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRCorrector(root)
    root.mainloop()