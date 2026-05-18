# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import pytesseract
import time
import os

# Укажите путь к tesseract
pytesseract.pytesseract.tesseract_cmd = r'D:\tesseract\tesseract.exe'

class OCRCorrector:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Grid Pro + Filters")
        self.root.geometry("1600x900")
        self.root.configure(bg='#1e1e1e')
        
        self.master_original = None # Самый первый файл
        self.original_img = None    # Текущий вход для режима (левое фото)
        self.processed_img = None   # Текущий результат режима (правое фото)
        
        self.left_photo = None
        self.right_photo = None
        
        self.points = []
        self.base_points = []
        self.triangles = []
        
        # Стеки истории изменений для отмены и повтора (Режим сетки)
        self.undo_stack = []
        self.redo_stack = []
        
        self.cell_angles = {}
        self.selected_cell = None
        self.dragging_idx = None
        self.dragging_group = []
        self.selected_points = set()       # Множество индексов выбранных точек
        self.first_shift_point = None      # Точка начала диапазона для Shift
        self.last_mouse_x, self.last_mouse_y = 0, 0
        
        self.grid_size_var = tk.StringVar(value="4")
        self.grid_mode_var = tk.StringVar(value="Треугольники")
        self.ocr_lang_var = tk.StringVar(value="rus") 
        
        self.filter_type_var = tk.StringVar(value="Оригинал")
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.IntVar(value=0)
        
        self.perspective_corners = None
        self.resize_corners = None
        
        self.image_offset_x, self.image_offset_y = 0, 0
        self.current_scale = 1.0
        
        self.setup_ui()
        self.bind_global_events()

    def validate_numeric(self, P):
        return P == "" or P.isdigit()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Настройка кастомного стиля для круглых/квадратных кнопок-иконок
        style.configure('IconButton.TButton', font=('Segoe UI', 12, 'bold'), width=3, padding=2)
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        top_panel = ttk.Frame(main_frame)
        top_panel.pack(fill=tk.X, pady=(0,10))
        
        # --- ЛЕВАЯ ЧАСТЬ ПАНЕЛИ (Основные действия и настройки) ---
        ttk.Button(top_panel, text="📁 Загрузить", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_panel, text="📝 OCR + Save", command=self.run_ocr).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_panel, text="💾 Сохранить фото", command=self.save_image).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(top_panel, text="Язык:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Combobox(top_panel, textvariable=self.ocr_lang_var, values=["rus", "eng"], width=10, state="readonly").pack(side=tk.LEFT, padx=5)
        
        ttk.Label(top_panel, text="Сетка:").pack(side=tk.LEFT, padx=(20,5))
        vcmd = (self.root.register(self.validate_numeric), '%P')
        self.grid_spin = ttk.Spinbox(top_panel, from_=1, to=30, width=5, textvariable=self.grid_size_var, validate='key', validatecommand=vcmd)
        self.grid_spin.pack(side=tk.LEFT, padx=5)
        self.grid_size_var.trace_add("write", lambda *args: self.on_grid_settings_change())
        
        self.mode_combo = ttk.Combobox(top_panel, textvariable=self.grid_mode_var, values=["Треугольники", "Квадраты"], width=12, state="readonly")
        self.mode_combo.pack(side=tk.LEFT, padx=5)
        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self.on_grid_settings_change())

        btn_redo = ttk.Button(top_panel, text="->", style='IconButton.TButton', command=self.redo)
        btn_redo.pack(side=tk.RIGHT, padx=2)
        ttk.ToolTip(btn_redo, "Повторить (Ctrl+Y)") if hasattr(self, 'ToolTip') else None

        btn_undo = ttk.Button(top_panel, text="<-", style='IconButton.TButton', command=self.undo)
        btn_undo.pack(side=tk.RIGHT, padx=2)

        btn_reset = ttk.Button(top_panel, text="🔄", style='IconButton.TButton', command=self.hard_reset)
        btn_reset.pack(side=tk.RIGHT, padx=(20, 2))

        filter_panel = ttk.LabelFrame(main_frame, text="🛠 Фильтры")
        filter_panel.pack(fill=tk.X, pady=(5,0))

        ttk.Label(filter_panel, text="Фильтр:").pack(side=tk.LEFT, padx=5)
        self.filter_combo = ttk.Combobox(filter_panel, textvariable=self.filter_type_var, values=["Оригинал", "Ч/Б Адаптив", "Инверсия", "Резкость", "CLAHE Контраст"], state="readonly", width=15)
        self.filter_combo.pack(side=tk.LEFT, padx=5)
        self.filter_combo.bind("<<ComboboxSelected>>", lambda e: self.update_displays())

        ttk.Label(filter_panel, text="Яркость:").pack(side=tk.LEFT, padx=(15, 5))
        ttk.Scale(filter_panel, from_=0.5, to=2.5, variable=self.brightness_var, orient=tk.HORIZONTAL, length=100, command=lambda e: self.update_displays()).pack(side=tk.LEFT, padx=5)

        ttk.Label(filter_panel, text="Контраст:").pack(side=tk.LEFT, padx=(15, 5))
        ttk.Scale(filter_panel, from_=-100, to=100, variable=self.contrast_var, orient=tk.HORIZONTAL, length=100, command=lambda e: self.update_displays()).pack(side=tk.LEFT, padx=5)
        
        mode_frame = ttk.LabelFrame(main_frame, text="Режим трансформации")
        mode_frame.pack(fill=tk.X, pady=(5,0))
        
        self.mode_var = tk.IntVar(value=0)
        modes = [("🔳 Сетка", 0), ("📐 Перспектива", 1), ("🎛 Геометрия", 2), ("🔄 Ротация ячеек", 3)]
        for text, mode_id in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, value=mode_id, command=self.on_mode_change).pack(side=tk.LEFT, padx=15, pady=5)

        img_frame = ttk.Frame(main_frame)
        img_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        left_frame = ttk.LabelFrame(img_frame, text="📄 Редактор")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
        self.left_canvas = tk.Canvas(left_frame, bg='#2b2b2b', highlightthickness=0)
        self.left_canvas.pack(fill=tk.BOTH, expand=True)
        
        right_frame = ttk.LabelFrame(img_frame, text="✅ Результат этапа")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5,0))
        self.right_canvas = tk.Canvas(right_frame, bg='#2b2b2b', highlightthickness=0)
        self.right_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.left_canvas.bind("<Button-1>", self.on_click)
        self.left_canvas.bind("<Button-3>", self.on_right_click)
        self.left_canvas.bind("<B1-Motion>", self.on_drag)
        self.left_canvas.bind("<ButtonRelease-1>", self.on_release)
        self.left_canvas.bind("<MouseWheel>", self.on_wheel_scaling)
        self.left_canvas.bind("<Configure>", lambda e: self.update_displays())

    def bind_global_events(self):
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        self.root.bind("<Control-a>", lambda e: self.select_all())

    # --- Логика Undo / Redo историй изменений ---
    def save_state(self):
        if self.mode_var.get() == 0 and self.points:
            self.undo_stack.append([p[:] for p in self.points])
            if len(self.undo_stack) > 50: 
                self.undo_stack.pop(0)
            self.redo_stack.clear()

    def undo(self):
        if self.mode_var.get() == 0 and self.undo_stack:
            self.redo_stack.append([p[:] for p in self.points])
            self.points = self.undo_stack.pop()
            self.process_warp(self.original_img, cv2.INTER_LINEAR)
            self.update_displays()

    def redo(self):
        if self.mode_var.get() == 0 and self.redo_stack:
            self.undo_stack.append([p[:] for p in self.points])
            self.points = self.redo_stack.pop()
            self.process_warp(self.original_img, cv2.INTER_LINEAR)
            self.update_displays()

    def select_all(self):
        if self.mode_var.get() == 0 and self.points:
            self.selected_points = set(range(len(self.points)))
            self.update_displays()

    def canvas_to_img(self, x, y):
        return ((x - self.image_offset_x) / self.current_scale, 
                (y - self.image_offset_y) / self.current_scale)

    def apply_filters(self, img):
        if img is None: return None
        res = img.copy()
        res = cv2.convertScaleAbs(res, alpha=self.brightness_var.get(), beta=self.contrast_var.get())
        f_type = self.filter_type_var.get()
        if f_type == "Ч/Б Адаптив":
            gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
            res = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
        elif f_type == "Инверсия": res = cv2.bitwise_not(res)
        elif f_type == "Резкость":
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            res = cv2.filter2D(res, -1, kernel)
        elif f_type == "CLAHE Контраст":
            lab = cv2.cvtColor(res, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            res = cv2.merge((clahe.apply(l), a, b))
            res = cv2.cvtColor(res, cv2.COLOR_LAB2RGB)
        return res

    def get_final_image(self):
        if self.processed_img is None: return None
        return self.apply_filters(self.processed_img)

    def on_mode_change(self):
        if self.original_img is None: return
        
        # Перенос результата в качестве нового исходника
        if self.processed_img is not None:
            self.original_img = self.processed_img.copy()
            
        mode = self.mode_var.get()
        self.selected_points.clear()
        self.cell_angles.clear()
        self.selected_cell = None
        self.undo_stack.clear()
        self.redo_stack.clear()

        if mode == 0:
            self.grid_spin.state(['!disabled'])
            self.init_warp_grid()
        elif mode == 1:
            self.grid_spin.state(['disabled'])
            self.init_perspective()
        elif mode == 2:
            self.grid_spin.state(['disabled'])
            self.init_geometry()
        elif mode == 3:
            self.grid_spin.state(['!disabled'])
            self.apply_cell_rotations()
            
        self.update_displays()
        self.root.focus_set()

    def init_warp_grid(self):
        if self.original_img is None: return
        h, w = self.original_img.shape[:2]
        try:
            n = int(self.grid_size_var.get())
        except: n = 4
        self.points = [[float(x), float(y)] for y in np.linspace(0, h, n + 1) for x in np.linspace(0, w, n + 1)]
        self.base_points = [p[:] for p in self.points]
        self.triangles = []
        cols = n + 1
        for j in range(n):
            for i in range(n):
                p1, p2, p3, p4 = j*cols+i, j*cols+i+1, (j+1)*cols+i, (j+1)*cols+i+1
                if self.grid_mode_var.get() == "Треугольники":
                    self.triangles.append((p1,p2,p3))
                    self.triangles.append((p2,p4,p3))
                else:
                    self.triangles.append((p1, p2, p4, p3))
        self.process_warp(self.original_img, cv2.INTER_LINEAR)

    def process_warp(self, base_img, quality):
        if not self.points: return
        h, w = base_img.shape[:2]
        out_img = np.zeros_like(base_img)
        for cell in self.triangles:
            src_pts = np.float32([self.points[i] for i in cell])
            dst_pts = np.float32([self.base_points[i] for i in cell])
            x, y, wb, hb = cv2.boundingRect(dst_pts)
            mask = np.zeros((hb, wb, 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(dst_pts - (x, y)), (1.0, 1.0, 1.0), 16, 0)
            sx, sy, sw, sh = cv2.boundingRect(src_pts)
            patch = base_img[max(0, sy):min(h, sy+sh), max(0, sx):min(w, sx+sw)]
            if patch.size == 0: continue
            if len(cell) == 3:
                M = cv2.getAffineTransform(np.float32(src_pts-(sx,sy))[:3], np.float32(dst_pts-(x,y))[:3])
                warped = cv2.warpAffine(patch, M, (wb, hb), flags=quality)
            else:
                M = cv2.getPerspectiveTransform(np.float32(src_pts-(sx,sy)), np.float32(dst_pts-(x,y)))
                warped = cv2.warpPerspective(patch, M, (wb, hb), flags=quality)
            view_y, view_x = slice(y, min(y+hb, h)), slice(x, min(x+wb, w))
            ph, pw = out_img[view_y, view_x].shape[:2]
            out_img[view_y, view_x] = (out_img[view_y, view_x]*(1-mask[:ph,:pw]) + warped[:ph,:pw]*mask[:ph,:pw]).astype(np.uint8)
        self.processed_img = out_img

    def apply_cell_rotations(self):
        if self.original_img is None: return
        base = self.original_img
        h, w = base.shape[:2]
        try: n = int(self.grid_size_var.get())
        except: n = 4
        out_img, cw, ch = base.copy(), w / n, h / n
        for (r, c), angle in self.cell_angles.items():
            if angle == 0: continue
            y1, y2, x1, x2 = int(r*ch), int((r+1)*ch), int(c*cw), int((c+1)*cw)
            roi = base[y1:y2, x1:x2]
            if angle == 90: roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180: roi = cv2.rotate(roi, cv2.ROTATE_180)
            elif angle == 270: roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
            out_img[y1:y2, x1:x2] = cv2.resize(roi, (x2-x1, y2-y1))
        self.processed_img = out_img

    def init_perspective(self):
        h, w = self.original_img.shape[:2]
        self.perspective_corners = np.array([[0,0], [w,0], [w,h], [0,h]], dtype=np.float32)
        self.apply_perspective_transform()

    def apply_perspective_transform(self):
        base = self.original_img
        h, w = base.shape[:2]
        M = cv2.getPerspectiveTransform(self.perspective_corners, np.float32([[0,0], [w,0], [w,h], [0,h]]))
        self.processed_img = cv2.warpPerspective(base, M, (w, h))

    def init_geometry(self):
        h, w = self.original_img.shape[:2]
        self.resize_corners = np.array([[0,0], [w,0], [w,h], [0,h]], dtype=np.float32)
        self.apply_geometric_transform()

    def apply_geometric_transform(self):
        base = self.original_img
        h, w = base.shape[:2]
        M = cv2.getPerspectiveTransform(np.float32([[0,0], [w,0], [w,h], [0,h]]), self.resize_corners)
        self.processed_img = cv2.warpPerspective(base, M, (w, h), borderValue=(255,255,255))

    def on_grid_settings_change(self):
        if self.original_img is None: return
        val = self.grid_size_var.get()
        if not val.isdigit(): return
        mode = self.mode_var.get()
        if mode == 0: self.init_warp_grid()
        elif mode == 3: self.apply_cell_rotations()
        self.update_displays()

    def on_click(self, event):
        if self.original_img is None: return
        self.left_canvas.focus_set()
        self.dragging_idx, self.dragging_group = None, []
        mode, scale, ox, oy = self.mode_var.get(), self.current_scale, self.image_offset_x, self.image_offset_y
        PT_TOL = 15
        
        ctrl = bool(event.state & 0x0004) or bool(event.state & 0x0080) # Windows / Linux Ctrl masks
        shift = bool(event.state & 0x0001)

        if mode == 0:
            self.save_state()
            mx, my = self.canvas_to_img(event.x, event.y)
            
            # 1. Поиск клика по узлу (точке)
            clicked_idx = next((i for i, p in enumerate(self.points) 
                                if np.hypot(p[0]-mx, p[1]-my) < PT_TOL/scale), None)
            
            if clicked_idx is not None:
                if ctrl:
                    if clicked_idx in self.selected_points:
                        self.selected_points.remove(clicked_idx)
                    else:
                        self.selected_points.add(clicked_idx)
                elif shift and self.first_shift_point is not None:
                    start, end = min(self.first_shift_point, clicked_idx), max(self.first_shift_point, clicked_idx)
                    for i in range(start, end + 1):
                        self.selected_points.add(i)
                else:
                    if clicked_idx not in self.selected_points:
                        self.selected_points = {clicked_idx}
                self.first_shift_point = clicked_idx
                self.dragging_idx = clicked_idx
                self.update_displays()
                return

            # 2. Поиск клика по грани (ребру линии)
            edge_found = False
            for cell in self.triangles:
                num_pts = len(cell)
                for i in range(num_pts):
                    p1_idx, p2_idx = cell[i], cell[(i+1)%num_pts]
                    p1, p2 = np.array(self.points[p1_idx]), np.array(self.points[p2_idx])
                    
                    line_vec = p2 - p1
                    p_vec = np.array([mx, my]) - p1
                    line_len = np.linalg.norm(line_vec)
                    if line_len == 0: continue
                    
                    unit_line = line_vec / line_len
                    projection = np.dot(p_vec, unit_line)
                    
                    if 0 <= projection <= line_len:
                        dist = np.linalg.norm(p_vec - projection * unit_line)
                        if dist < 8 / scale:
                            if ctrl:
                                self.selected_points.update([p1_idx, p2_idx])
                            else:
                                self.selected_points = {p1_idx, p2_idx}
                            edge_found = True
                            self.dragging_group = list(self.selected_points)
                            self.last_mouse_x, self.last_mouse_y = event.x, event.y
                            break
                if edge_found: break
            
            if edge_found:
                self.update_displays()
                return

            # 3. Поиск клика внутри полигона (ячейки)
            for cell in self.triangles:
                poly = [[self.points[i][0]*scale + ox, self.points[i][1]*scale + oy] for i in cell]
                if cv2.pointPolygonTest(np.array(poly, dtype=np.int32), (event.x, event.y), False) >= 0:
                    if ctrl:
                        self.selected_points.update(cell)
                    else:
                        self.selected_points = set(cell)
                    self.dragging_group = list(self.selected_points)
                    self.last_mouse_x, self.last_mouse_y = event.x, event.y
                    self.update_displays()
                    return
            
            # Клик мимо всего — сбрасываем выделение, если Ctrl/Shift не зажаты
            if not (ctrl or shift):
                self.selected_points.clear()
                self.update_displays()

        elif mode == 3:
            nx, ny = (event.x - ox) / scale, (event.y - oy) / scale
            try: n = int(self.grid_size_var.get())
            except: n = 4
            h, w = self.original_img.shape[:2]
            if 0 <= nx <= w and 0 <= ny <= h:
                self.selected_cell = (int(ny // (h/n)), int(nx // (w/n)))
                self.update_displays()
        else:
            target = self.perspective_corners if mode == 1 else self.resize_corners
            for i, p in enumerate(target):
                if abs(p[0]*scale + ox - event.x) < PT_TOL*1.5 and abs(p[1]*scale + oy - event.y) < PT_TOL*1.5:
                    self.dragging_idx = i
                    return

    def on_drag(self, event):
        if self.original_img is None: return
        scale, h, w = self.current_scale, *self.original_img.shape[:2]
        nx = max(0, min(w, (event.x - self.image_offset_x) / scale))
        ny = max(0, min(h, (event.y - self.image_offset_y) / scale))

        if self.dragging_idx is not None:
            if self.mode_var.get() == 0:
                dx, dy = nx - self.points[self.dragging_idx][0], ny - self.points[self.dragging_idx][1]
                for idx in self.selected_points:
                    self.points[idx][0] = np.clip(self.points[idx][0] + dx, 0, w)
                    self.points[idx][1] = np.clip(self.points[idx][1] + dy, 0, h)
                self.process_warp(self.original_img, cv2.INTER_NEAREST)
            elif self.mode_var.get() == 1:
                self.perspective_corners[self.dragging_idx] = [nx, ny]
                self.apply_perspective_transform()
            elif self.mode_var.get() == 2:
                self.resize_corners[self.dragging_idx] = [nx, ny]
                self.apply_geometric_transform()
        elif self.dragging_group:
            dx, dy = (event.x - self.last_mouse_x) / scale, (event.y - self.last_mouse_y) / scale
            target_pts = self.selected_points if self.selected_points else self.dragging_group
            for idx in target_pts:
                self.points[idx][0] = np.clip(self.points[idx][0] + dx, 0, w)
                self.points[idx][1] = np.clip(self.points[idx][1] + dy, 0, h)
            self.last_mouse_x, self.last_mouse_y = event.x, event.y
            self.process_warp(self.original_img, cv2.INTER_NEAREST)
        self.update_displays()

    def on_release(self, event):
        if self.mode_var.get() == 0: 
            self.process_warp(self.original_img, cv2.INTER_LINEAR)
        self.dragging_idx, self.dragging_group = None, []
        self.update_displays()

    def on_wheel_scaling(self, event):
        # Проверяем, зажат ли Alt (стандартные маски под разные ОС)
        is_alt_pressed = (event.state & 0x0020) != 0 or (event.state & 131072) != 0
        if not is_alt_pressed or not self.selected_points or self.mode_var.get() != 0: 
            return
        
        self.save_state()
        k = 1.05 if event.delta > 0 else 0.95
        
        # Вычисляем геометрический центр текущей группы выделенных точек
        pts_coords = np.array([self.points[i] for i in self.selected_points])
        center = np.mean(pts_coords, axis=0)
        
        h, w = self.original_img.shape[:2]
        for i in self.selected_points:
            p = np.array(self.points[i])
            new_p = center + (p - center) * k
            self.points[i] = [np.clip(new_p[0], 0, w), np.clip(new_p[1], 0, h)]
            
        self.process_warp(self.original_img, cv2.INTER_LINEAR)
        self.update_displays()

    def on_key_press(self, event):
        if self.mode_var.get() == 3 and self.selected_cell:
            key = event.keysym.lower()
            if key in ['a', 'ф']: self.cell_angles[self.selected_cell] = (self.cell_angles.get(self.selected_cell, 0) - 90) % 360
            elif key in ['d', 'в']: self.cell_angles[self.selected_cell] = (self.cell_angles.get(self.selected_cell, 0) + 90) % 360
            self.apply_cell_rotations()
            self.update_displays()

    def update_displays(self):
        if self.original_img is None: return
        cw, ch = self.left_canvas.winfo_width(), self.left_canvas.winfo_height()
        if cw < 10: return
        h, w = self.original_img.shape[:2]
        self.current_scale = min(cw/w, ch/h) * 0.95
        sw, sh = int(w * self.current_scale), int(h * self.current_scale)
        self.image_offset_x, self.image_offset_y = (cw - sw) // 2, (ch - sh) // 2
        
        self.left_photo = ImageTk.PhotoImage(Image.fromarray(cv2.resize(self.original_img, (sw, sh))))
        self.left_canvas.delete("all")
        self.left_canvas.create_image(cw//2, ch//2, image=self.left_photo)
        self.draw_overlay()
        
        if self.processed_img is not None:
            final = self.get_final_image()
            self.right_photo = ImageTk.PhotoImage(Image.fromarray(cv2.resize(final, (sw, sh))))
            self.right_canvas.delete("all")
            self.right_canvas.create_image(self.right_canvas.winfo_width()//2, self.right_canvas.winfo_height()//2, image=self.right_photo)

    def draw_overlay(self):
        m, s, ox, oy = self.mode_var.get(), self.current_scale, self.image_offset_x, self.image_offset_y
        if m == 0:
            for c in self.triangles:
                pts = []
                for i in c: pts.extend([self.points[i][0]*s+ox, self.points[i][1]*s+oy])
                self.left_canvas.create_polygon(pts, fill="", outline="#00ffcc", width=1)
            for i, p in enumerate(self.points):
                px, py = p[0]*s+ox, p[1]*s+oy
                is_sel = i in self.selected_points
                r = 5 if is_sel else 3
                color = "#ffcc00" if is_sel else "#ff3366"
                self.left_canvas.create_oval(px-r, py-r, px+r, py+r, fill=color, outline="white" if is_sel else "")
        elif m == 1:
            pts = [(p[0]*s+ox, p[1]*s+oy) for p in self.perspective_corners]
            self.left_canvas.create_polygon(pts, fill="", outline="#ffaa00", width=2)
            for p in self.perspective_corners:
                px, py = p[0]*s+ox, p[1]*s+oy
                self.left_canvas.create_oval(px-5, py-5, px+5, py+5, fill="#ffaa00", outline="white")
        elif m == 2:
            pts = [(p[0]*s+ox, p[1]*s+oy) for p in self.resize_corners]
            self.left_canvas.create_polygon(pts, fill="", outline="#00aaff", width=2)
            for p in self.resize_corners:
                px, py = p[0]*s+ox, p[1]*s+oy
                self.left_canvas.create_oval(px-5, py-5, px+5, py+5, fill="#00aaff", outline="white")
        elif m == 3:
            try: n = int(self.grid_size_var.get())
            except: n = 4
            h_i, w_i = self.original_img.shape[:2]
            for i in range(n + 1):
                self.left_canvas.create_line(ox, i*(h_i/n)*s+oy, w_i*s+ox, i*(h_i/n)*s+oy, fill="#555")
                self.left_canvas.create_line(i*(w_i/n)*s+ox, oy, i*(w_i/n)*s+ox, h_i*s+oy, fill="#555")
            if self.selected_cell:
                r, c = self.selected_cell
                self.left_canvas.create_rectangle(c*(w_i/n)*s+ox, r*(h_i/n)*s+oy, (c+1)*(w_i/n)*s+ox, (r+1)*(h_i/n)*s+oy, outline="#ff3366", width=3)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path)
            if img is None: return
            self.master_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.hard_reset()

    def hard_reset(self):
        if self.master_original is not None:
            self.original_img = self.master_original.copy()
            self.processed_img = self.original_img.copy()
            self.selected_points.clear()
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.mode_var.set(0)
            self.on_mode_change()

    def on_right_click(self, event):
        if self.mode_var.get() == 0:
            self.selected_points.clear()
            self.update_displays()

    def save_image(self):
        final = self.get_final_image()
        if final is None: return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path: cv2.imwrite(path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

    def run_ocr(self):
        final = self.get_final_image()
        if final is None: return
        save_path = filedialog.asksaveasfilename(defaultextension=".txt")
        if not save_path: return
        try:
            text = pytesseract.image_to_string(cv2.cvtColor(final, cv2.COLOR_RGB2GRAY), lang=self.ocr_lang_var.get())
            with open(save_path, "w", encoding="utf-8") as f: f.write(text)
            cv2.imwrite(os.path.splitext(save_path)[0] + ".png", cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Готово", "Текст и фото сохранены!")
        except Exception as e: messagebox.showerror("Ошибка", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRCorrector(root)
    root.mainloop()