# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import pytesseract
import time
import os

# Укажите путь к tesseract, если он отличается
pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'

class OCRCorrector:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Grid v3.4 PRO - Fixed Input & Focus")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1e1e1e')
        
        self.original_img = None
        self.processed_img = None
        self.left_photo = None
        self.right_photo = None
        
        # Данные деформации (Warp)
        self.points = []
        self.base_points = []
        self.triangles = []
        
        # Независимые настройки сетки
        self.warp_grid_size = 4
        self.rotation_grid_size = 4
        
        # Данные ротации ячеек
        self.cell_angles = {} 
        self.selected_cell = None 
        
        # Состояние мыши
        self.dragging_idx = None
        self.dragging_group = []
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.last_update_time = 0
        
        self.current_mode = 0  
        self.grid_size_var = tk.StringVar(value="4") # Используем StringVar для удобной валидации
        self.grid_mode_var = tk.StringVar(value="Треугольники")
        
        self.perspective_corners = None
        self.resize_corners = None
        
        self.image_offset_x = 0
        self.image_offset_y = 0
        self.current_scale = 1.0
        self.current_base_img = None
        
        self.setup_ui()
        self.root.bind("<KeyPress>", self.on_key_press)

    def validate_numeric(self, P):
        """Разрешает только цифры или пустую строку в инпуте"""
        if P == "" or P.isdigit():
            return True
        return False

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # --- ВЕРХНЯЯ ПАНЕЛЬ ---
        top_panel = ttk.Frame(main_frame)
        top_panel.pack(fill=tk.X, pady=(0,10))
        
        ttk.Button(top_panel, text="📁 Загрузить", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_panel, text="🔄 Сброс", command=self.reset_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_panel, text="📝 OCR + Save", command=self.run_ocr).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_panel, text="💾 Сохранить фото", command=self.save_image).pack(side=tk.LEFT, padx=5)
        
        # Контроль сетки с валидацией
        ttk.Label(top_panel, text="Размер сетки:").pack(side=tk.LEFT, padx=(20,5))
        
        vcmd = (self.root.register(self.validate_numeric), '%P')
        self.grid_spin = ttk.Spinbox(top_panel, from_=1, to=30, width=5, 
                                     textvariable=self.grid_size_var, 
                                     command=self.on_grid_settings_change,
                                     validate='key', validatecommand=vcmd)
        self.grid_spin.pack(side=tk.LEFT, padx=5)
        
        # Отслеживание изменений
        self.grid_size_var.trace_add("write", lambda *args: self.on_grid_settings_change())
        
        self.mode_combo = ttk.Combobox(top_panel, textvariable=self.grid_mode_var, 
                                       values=["Треугольники", "Квадраты"], width=12, state="readonly")
        self.mode_combo.pack(side=tk.LEFT, padx=5)
        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self.on_grid_settings_change())
        
        # --- ПАНЕЛЬ РЕЖИМОВ ---
        mode_frame = ttk.LabelFrame(main_frame, text="Режим управления")
        mode_frame.pack(fill=tk.X, pady=(10,0))
        
        self.mode_var = tk.IntVar(value=0)
        modes = [
            ("🔳 Сетка (Деформация)", 0), 
            ("📐 Перспектива", 1), 
            ("🎛 Геометрия", 2), 
            ("🔄 Ротация ячеек", 3)
        ]
        
        for text, mode_id in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, 
                          value=mode_id, command=self.on_mode_change).pack(side=tk.LEFT, padx=15, pady=5)
        
        # --- ОБЛАСТЬ ИЗОБРАЖЕНИЙ ---
        img_frame = ttk.Frame(main_frame)
        img_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        left_frame = ttk.LabelFrame(img_frame, text="📄 Редактор (Интерактив)")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))
        self.left_canvas = tk.Canvas(left_frame, bg='#2b2b2b', highlightthickness=0)
        self.left_canvas.pack(fill=tk.BOTH, expand=True)
        
        right_frame = ttk.LabelFrame(img_frame, text="✅ Результат")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10,0))
        self.right_canvas = tk.Canvas(right_frame, bg='#2b2b2b', highlightthickness=0)
        self.right_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.left_canvas.bind("<Button-1>", self.on_click)
        self.left_canvas.bind("<B1-Motion>", self.on_drag)
        self.left_canvas.bind("<ButtonRelease-1>", self.on_release)
        self.left_canvas.bind("<Configure>", lambda e: self.update_displays())

    def on_grid_settings_change(self):
        if self.original_img is None: return
        raw_val = self.grid_size_var.get()
        if not raw_val or not raw_val.isdigit(): return # Игнорируем пустой ввод
        
        val = int(raw_val)
        mode = self.mode_var.get()
        
        if mode == 0:
            self.warp_grid_size = val
            self.init_warp_grid()
        elif mode == 3:
            self.rotation_grid_size = val
            self.cell_angles.clear()
            self.selected_cell = None
            self.apply_cell_rotations()
        
        self.update_displays()

    def on_mode_change(self):
        if self.processed_img is not None:
            self.current_base_img = self.processed_img.copy()

        self.current_mode = self.mode_var.get()
        
        if self.current_mode == 0:
            self.grid_spin.state(['!disabled'])
            self.mode_combo.state(['!disabled'])
            self.grid_size_var.set(str(self.warp_grid_size))
            self.init_warp_grid()
        elif self.current_mode == 3:
            self.grid_spin.state(['!disabled'])
            self.mode_combo.state(['disabled'])
            self.grid_size_var.set(str(self.rotation_grid_size))
            self.cell_angles.clear()
            self.selected_cell = None
            self.apply_cell_rotations()
        else:
            self.grid_spin.state(['disabled'])
            self.mode_combo.state(['disabled'])
            if self.current_mode == 1: self.init_perspective()
            elif self.current_mode == 2: self.init_geometry()
            
        self.update_displays()
        # При смене режима убираем фокус с инпутов
        self.root.focus_set()

    def on_click(self, event):
        # Клик по холсту забирает фокус у Spinbox, чтобы работали A/D
        self.left_canvas.focus_set()
        
        if self.original_img is None: return
        self.dragging_idx = None
        self.dragging_group = []
        
        mode = self.mode_var.get()
        scale = self.current_scale
        ox, oy = self.image_offset_x, self.image_offset_y
        
        if mode == 3:
            nx, ny = (event.x - ox) / scale, (event.y - oy) / scale
            n = self.rotation_grid_size
            h, w = self.original_img.shape[:2]
            if 0 <= nx <= w and 0 <= ny <= h:
                col, row = int(nx // (w/n)), int(ny // (h/n))
                self.selected_cell = (min(row, n-1), min(col, n-1))
                self.update_displays()
            return

        PT_TOL = 15
        if mode == 0: 
            for i, p in enumerate(self.points):
                sx, sy = p[0]*scale + ox, p[1]*scale + oy
                if abs(sx - event.x) < PT_TOL and abs(sy - event.y) < PT_TOL:
                    self.dragging_idx = i; return
        elif mode in [1, 2]:
            target = self.perspective_corners if mode == 1 else self.resize_corners
            for i, p in enumerate(target):
                if abs(p[0]*scale + ox - event.x) < PT_TOL*1.5 and abs(p[1]*scale + oy - event.y) < PT_TOL*1.5:
                    self.dragging_idx = i; return

    def on_key_press(self, event):
        # Если фокус сейчас в поле ввода, не обрабатываем горячие клавиши поворота
        if self.root.focus_get() == self.grid_spin:
            return

        if self.mode_var.get() != 3 or not self.selected_cell: return
        key = event.keysym.lower()
        if key in ['a', 'ф']:
            self.cell_angles[self.selected_cell] = (self.cell_angles.get(self.selected_cell, 0) - 90) % 360
            self.apply_cell_rotations(); self.update_displays()
        elif key in ['d', 'в']:
            self.cell_angles[self.selected_cell] = (self.cell_angles.get(self.selected_cell, 0) + 90) % 360
            self.apply_cell_rotations(); self.update_displays()

    # --- ОСТАЛЬНАЯ ЛОГИКА (без изменений, но включена для работы) ---

    def init_warp_grid(self):
        if self.original_img is None: return
        base = self.current_base_img if self.current_base_img is not None else self.original_img
        h, w = base.shape[:2]
        n = max(1, self.warp_grid_size)
        self.points = [[float(x), float(y)] for y in np.linspace(0, h, n + 1) for x in np.linspace(0, w, n + 1)]
        self.base_points = [p[:] for p in self.points]
        self.triangles = []
        cols = n + 1
        for j in range(n):
            for i in range(n):
                p1, p2, p3, p4 = j*cols+i, j*cols+i+1, (j+1)*cols+i, (j+1)*cols+i+1
                if self.grid_mode_var.get() == "Треугольники":
                    self.triangles.append((p1, p2, p3)); self.triangles.append((p2, p4, p3))
                else: self.triangles.append((p1, p2, p4, p3))
        self.process_warp(base, cv2.INTER_LINEAR)

    def init_perspective(self):
        if self.original_img is None: return
        base = self.current_base_img if self.current_base_img is not None else self.original_img
        h, w = base.shape[:2]
        self.perspective_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        self.apply_perspective_transform()

    def apply_perspective_transform(self):
        base = self.current_base_img if self.current_base_img is not None else self.original_img
        h, w = base.shape[:2]
        M = cv2.getPerspectiveTransform(self.perspective_corners, np.float32([[0,0], [w,0], [w,h], [0,h]]))
        self.processed_img = cv2.warpPerspective(base, M, (w, h))

    def init_geometry(self):
        if self.original_img is None: return
        h, w = self.original_img.shape[:2]
        self.resize_corners = np.array([[0,0], [w,0], [w,h], [0,h]], dtype=np.float32)
        self.apply_geometric_transform()

    def apply_geometric_transform(self):
        base = self.current_base_img if self.current_base_img is not None else self.original_img
        h, w = base.shape[:2]
        M = cv2.getPerspectiveTransform(np.float32([[0,0], [w,0], [w,h], [0,h]]), self.resize_corners)
        self.processed_img = cv2.warpPerspective(base, M, (w, h), borderValue=(255,255,255))

    def on_drag(self, event):
        if self.original_img is None or self.mode_var.get() == 3: return
        if self.dragging_idx is None: return
        
        curr_time = time.time()
        if curr_time - self.last_update_time < 0.012: return 
        self.last_update_time = curr_time

        scale = self.current_scale
        h, w = self.original_img.shape[:2]
        base = self.current_base_img if self.current_base_img is not None else self.original_img

        nx, ny = max(0, min(w, (event.x - self.image_offset_x) / scale)), max(0, min(h, (event.y - self.image_offset_y) / scale))
        if self.current_mode == 0: 
            self.points[self.dragging_idx] = [nx, ny]
            self.process_warp(base, cv2.INTER_NEAREST)
        elif self.current_mode == 1: 
            self.perspective_corners[self.dragging_idx] = [nx, ny]
            self.apply_perspective_transform()
        elif self.current_mode == 2: 
            self.resize_corners[self.dragging_idx] = [nx, ny]
            self.apply_geometric_transform()
        
        self.update_displays()

    def on_release(self, event):
        if self.mode_var.get() == 0:
            self.process_warp(self.current_base_img if self.current_base_img is not None else self.original_img, cv2.INTER_LINEAR)
        self.dragging_idx = None
        self.update_displays()

    def apply_cell_rotations(self):
        base = self.current_base_img if self.current_base_img is not None else self.original_img
        if base is None: return
        out_img = base.copy()
        h, w = base.shape[:2]
        n = self.rotation_grid_size
        cw, ch = w / n, h / n
        for (r, c), angle in self.cell_angles.items():
            if angle == 0: continue
            y1, y2, x1, x2 = int(r*ch), int((r+1)*ch), int(c*cw), int((c+1)*cw)
            roi = base[y1:y2, x1:x2]
            if angle == 90: roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180: roi = cv2.rotate(roi, cv2.ROTATE_180)
            elif angle == 270: roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
            out_img[y1:y2, x1:x2] = cv2.resize(roi, (x2-x1, y2-y1))
        self.processed_img = out_img

    def process_warp(self, base_img, quality):
        if not self.points: return
        h, w = base_img.shape[:2]
        out_img = np.zeros_like(base_img)
        for cell in self.triangles:
            src_pts, dst_pts = np.float32([self.points[i] for i in cell]), np.float32([self.base_points[i] for i in cell])
            x, y, wb, hb = cv2.boundingRect(dst_pts)
            mask = np.zeros((hb, wb, 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(dst_pts - (x, y)), (1.0, 1.0, 1.0), 16, 0)
            sx, sy, sw, sh = cv2.boundingRect(src_pts)
            patch = base_img[max(0, sy):min(h, sy+sh), max(0, sx):min(w, sx+sw)]
            if patch.size == 0: continue
            M = cv2.getAffineTransform(np.float32(src_pts-(sx,sy))[:3], np.float32(dst_pts-(x,y))[:3]) if len(cell)==3 else cv2.getPerspectiveTransform(np.float32(src_pts-(sx,sy)), np.float32(dst_pts-(x,y)))
            warped = cv2.warpAffine(patch, M, (wb, hb), flags=quality, borderMode=cv2.BORDER_REFLECT_101) if len(cell)==3 else cv2.warpPerspective(patch, M, (wb, hb), flags=quality, borderMode=cv2.BORDER_REFLECT_101)
            view_y, view_x = slice(y, min(y+hb, h)), slice(x, min(x+wb, w))
            ph, pw = out_img[view_y, view_x].shape[:2]
            out_img[view_y, view_x] = (out_img[view_y, view_x]*(1-mask[:ph,:pw]) + warped[:ph,:pw]*mask[:ph,:pw]).astype(np.uint8)
        self.processed_img = out_img

    def update_displays(self):
        if self.original_img is None: return
        h, w = self.original_img.shape[:2]
        cw, ch = self.left_canvas.winfo_width(), self.left_canvas.winfo_height()
        if cw < 10: return
        self.current_scale = min(cw/w, ch/h) * 0.92
        sw, sh = int(w * self.current_scale), int(h * self.current_scale)
        self.image_offset_x, self.image_offset_y = (cw - sw) // 2, (ch - sh) // 2
        
        self.left_photo = ImageTk.PhotoImage(Image.fromarray(cv2.resize(self.original_img, (sw, sh))))
        self.left_canvas.delete("all")
        self.left_canvas.create_image(cw//2, ch//2, image=self.left_photo)
        self.draw_overlay()
        
        if self.processed_img is not None:
            self.right_photo = ImageTk.PhotoImage(Image.fromarray(cv2.resize(self.processed_img, (sw, sh))))
            self.right_canvas.delete("all")
            self.right_canvas.create_image(self.right_canvas.winfo_width()//2, self.right_canvas.winfo_height()//2, image=self.right_photo)

    def draw_overlay(self):
        m, s, ox, oy = self.mode_var.get(), self.current_scale, self.image_offset_x, self.image_offset_y
        if m == 0:
            for c in self.triangles:
                pts = []
                for i in c: pts.extend([self.points[i][0]*s+ox, self.points[i][1]*s+oy])
                self.left_canvas.create_polygon(pts, fill="", outline="#00ffcc", width=1)
        elif m == 1:
            pts = [(p[0]*s+ox, p[1]*s+oy) for p in self.perspective_corners]
            self.left_canvas.create_polygon(pts, fill="", outline="#ffaa00", width=2)
            for p in pts: self.left_canvas.create_oval(p[0]-6, p[1]-6, p[0]+6, p[1]+6, fill="#ffaa00")
        elif m == 2:
            pts = [(p[0]*s+ox, p[1]*s+oy) for p in self.resize_corners]
            self.left_canvas.create_polygon(pts, fill="", outline="#00aaff", width=2)
        elif m == 3:
            n = self.rotation_grid_size
            h, w = self.original_img.shape[:2]
            for i in range(n + 1):
                self.left_canvas.create_line(ox, i*(h/n)*s+oy, w*s+ox, i*(h/n)*s+oy, fill="#555")
                self.left_canvas.create_line(i*(w/n)*s+ox, oy, i*(w/n)*s+ox, h*s+oy, fill="#555")
            if self.selected_cell:
                r, c = self.selected_cell
                self.left_canvas.create_rectangle(c*(w/n)*s+ox, r*(h/n)*s+oy, (c+1)*(w/n)*s+ox, (r+1)*(h/n)*s+oy, outline="#ff3366", width=3)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path)
            self.original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.reset_all()

    def reset_all(self):
        self.current_base_img = None
        self.cell_angles.clear()
        self.selected_cell = None
        self.warp_grid_size = 4
        self.rotation_grid_size = 4
        self.grid_size_var.set("4")
        if self.original_img is not None:
            self.processed_img = self.original_img.copy()
            self.on_mode_change()
        self.update_displays()

    def save_image(self):
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path: cv2.imwrite(path, cv2.cvtColor(self.processed_img, cv2.COLOR_RGB2BGR))

    def run_ocr(self):
        if self.processed_img is None: return
        save_path = filedialog.asksaveasfilename(defaultextension=".txt")
        if not save_path: return
        try:
            gray = cv2.cvtColor(self.processed_img, cv2.COLOR_RGB2GRAY)
            text = pytesseract.image_to_string(gray, lang='eng+rus')
            with open(save_path, "w", encoding="utf-8") as f: f.write(text)
            cv2.imwrite(os.path.splitext(save_path)[0] + ".png", cv2.cvtColor(self.processed_img, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Готово", "Текст и фото сохранены!")
        except Exception as e: messagebox.showerror("Ошибка", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRCorrector(root)
    root.mainloop()