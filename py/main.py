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
        self.root.title("Устранение искажений сканированного текста (OCR Grid v3.0 PRO)")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1e1e1e')
        
        self.original_img = None
        self.processed_img = None
        self.left_photo = None
        self.right_photo = None
        
        # Данные сетки
        self.points = []
        self.base_points = []
        self.triangles = []
        
        # Состояние мыши
        self.dragging_idx = None
        self.dragging_group = []
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.last_update_time = 0
        
        self.current_mode = 0  
        self.grid_size_var = tk.IntVar(value=4)
        self.grid_mode_var = tk.StringVar(value="Треугольники")
        
        self.perspective_corners = None
        self.resize_corners = None
        
        self.image_offset_x = 0
        self.image_offset_y = 0
        self.current_scale = 1.0
        self.current_base_img = None
        
        self.setup_ui()
        self.root.after(500, self.show_welcome_alert)

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        top_panel = ttk.Frame(main_frame)
        top_panel.pack(fill=tk.X, pady=(0,10))
        
        ttk.Button(top_panel, text="📁 Загрузить", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_panel, text="🔄 Сброс", command=self.reset_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_panel, text="📝 OCR PRO + Save", command=self.run_ocr).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_panel, text="💾 Сохранить фото", command=self.save_image).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(top_panel, text="Сетка:").pack(side=tk.LEFT, padx=(20,5))
        self.grid_spin = ttk.Spinbox(top_panel, from_=1, to=7, width=5, textvariable=self.grid_size_var, command=self.init_grid)
        self.grid_spin.pack(side=tk.LEFT, padx=5)
        
        self.mode_combo = ttk.Combobox(top_panel, textvariable=self.grid_mode_var, 
                                     values=["Треугольники", "Квадраты"], width=12, state="readonly")
        self.mode_combo.pack(side=tk.LEFT, padx=5)
        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self.init_grid())
        
        mode_frame = ttk.LabelFrame(main_frame, text="Режим управления")
        mode_frame.pack(fill=tk.X, pady=(10,0))
        
        self.mode_var = tk.IntVar(value=0)
        modes = [("🔳 Сетка", 0), ("📐 Перспектива", 1), ("🎛 Геометрия", 2)]
        
        for text, mode_id in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, 
                          value=mode_id, command=self.on_mode_change).pack(side=tk.LEFT, padx=15, pady=5)
        
        img_frame = ttk.Frame(main_frame)
        img_frame.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.LabelFrame(img_frame, text="📄 Редактор (Точки / Грани / Ячейки)")
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

    def show_welcome_alert(self):
        messagebox.showinfo("🎉 OCR Grid v3.0", 
                          "Инструкция по Сетке:\n"
                          "• Тяните ТОЧКУ для локальной правки\n"
                          "• Тяните ЛИНИЮ для смещения грани\n"
                          "• Тяните ЦЕНТР ячейки для её перемещения")

    def on_click(self, event):
        if self.original_img is None: return
        self.dragging_idx = None
        self.dragging_group = []
        
        mode = self.mode_var.get()
        scale = self.current_scale
        ox, oy = self.image_offset_x, self.image_offset_y
        
        PT_TOL = 15 
        LN_TOL = 8

        if mode == 0: 
            for i, p in enumerate(self.points):
                sx, sy = p[0]*scale + ox, p[1]*scale + oy
                if abs(sx - event.x) < PT_TOL and abs(sy - event.y) < PT_TOL:
                    self.dragging_idx = i
                    return

            for cell in self.triangles:
                n = len(cell)
                for i in range(n):
                    idx1, idx2 = cell[i], cell[(i + 1) % n]
                    p1 = np.array([self.points[idx1][0]*scale + ox, self.points[idx1][1]*scale + oy])
                    p2 = np.array([self.points[idx2][0]*scale + ox, self.points[idx2][1]*scale + oy])
                    mouse = np.array([event.x, event.y])
                    line_vec = p2 - p1
                    line_len = np.linalg.norm(line_vec)
                    if line_len == 0: continue
                    t = max(0, min(1, np.dot(mouse - p1, line_vec) / (line_len**2)))
                    dist = np.linalg.norm(mouse - (p1 + t * line_vec))
                    if dist < LN_TOL:
                        self.dragging_group = [idx1, idx2]
                        self.last_mouse_x, self.last_mouse_y = event.x, event.y
                        return

            for cell in self.triangles:
                poly = [[self.points[i][0]*scale + ox, self.points[i][1]*scale + oy] for i in cell]
                if cv2.pointPolygonTest(np.array(poly, dtype=np.int32), (event.x, event.y), False) >= 0:
                    self.dragging_group = list(cell)
                    self.last_mouse_x, self.last_mouse_y = event.x, event.y
                    return
        
        else:
            target_pts = self.perspective_corners if mode == 1 else self.resize_corners
            if target_pts is not None:
                for i, p in enumerate(target_pts):
                    sx, sy = p[0]*scale + ox, p[1]*scale + oy
                    if abs(sx - event.x) < PT_TOL*1.5 and abs(sy - event.y) < PT_TOL*1.5:
                        self.dragging_idx = i
                        return

    def on_drag(self, event):
        if self.original_img is None: return
        if self.dragging_idx is None and not self.dragging_group: return
        
        curr_time = time.time()
        if curr_time - self.last_update_time < 0.016: return 
        self.last_update_time = curr_time

        scale = self.current_scale
        h, w = self.original_img.shape[:2]
        base_img = self.current_base_img if self.current_base_img is not None else self.original_img

        if self.dragging_idx is not None:
            nx = max(0, min(w, (event.x - self.image_offset_x) / scale))
            ny = max(0, min(h, (event.y - self.image_offset_y) / scale))
            
            if self.current_mode == 0:
                self.points[self.dragging_idx] = [nx, ny]
                self.process_warp(base_img, cv2.INTER_NEAREST)
            elif self.current_mode == 1:
                self.perspective_corners[self.dragging_idx] = [nx, ny]
                self.apply_perspective_transform()
            elif self.current_mode == 2:
                self.resize_corners[self.dragging_idx] = [nx, ny]
                self.apply_geometric_transform()
        
        elif self.dragging_group:
            dx = (event.x - self.last_mouse_x) / scale
            dy = (event.y - self.last_mouse_y) / scale
            for idx in self.dragging_group:
                self.points[idx][0] = max(0, min(w, self.points[idx][0] + dx))
                self.points[idx][1] = max(0, min(h, self.points[idx][1] + dy))
            self.last_mouse_x, self.last_mouse_y = event.x, event.y
            self.process_warp(base_img, cv2.INTER_NEAREST)
            
        self.update_displays()

    def on_release(self, event):
        self.dragging_idx = None
        self.dragging_group = []
        base_img = self.current_base_img if self.current_base_img is not None else self.original_img
        if self.current_mode == 0: 
            self.process_warp(base_img, cv2.INTER_LINEAR)
        self.update_displays()

    def process_warp(self, base_img, quality):
        if not self.points: return
        h, w = base_img.shape[:2]
        out_img = np.zeros_like(base_img)
        
        for cell in self.triangles:
            src_pts = np.float32([self.points[i] for i in cell])
            dst_pts = np.float32([self.base_points[i] for i in cell])
            
            x, y, wb, hb = cv2.boundingRect(dst_pts)
            dst_rect = dst_pts - (x, y)
            
            mask = np.zeros((hb, wb, 3), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(dst_rect), (1.0, 1.0, 1.0), 16, 0)
            
            sx, sy, sw, sh = cv2.boundingRect(src_pts)
            img_patch = base_img[max(0, sy):min(h, sy+sh), max(0, sx):min(w, sx+sw)]
            if img_patch.size == 0: continue
            src_in_patch = src_pts - (sx, sy)

            if len(cell) == 3:
                M = cv2.getAffineTransform(np.float32(src_in_patch[:3]), np.float32(dst_rect[:3]))
                warped = cv2.warpAffine(img_patch, M, (wb, hb), flags=quality, borderMode=cv2.BORDER_REFLECT_101)
            else:
                M = cv2.getPerspectiveTransform(np.float32(src_in_patch), np.float32(dst_rect))
                warped = cv2.warpPerspective(img_patch, M, (wb, hb), flags=quality, borderMode=cv2.BORDER_REFLECT_101)
            
            view_y, view_x = slice(y, min(y+hb, h)), slice(x, min(x+wb, w))
            ph, pw = out_img[view_y, view_x].shape[:2]
            out_img[view_y, view_x] = (out_img[view_y, view_x] * (1 - mask[:ph, :pw]) + 
                                       warped[:ph, :pw] * mask[:ph, :pw]).astype(np.uint8)
                                       
        self.processed_img = out_img

    def update_displays(self):
        if self.original_img is None: return
        h, w = self.original_img.shape[:2]
        cw, ch = self.left_canvas.winfo_width(), self.left_canvas.winfo_height()
        if cw < 10: return

        self.current_scale = min(cw/w, ch/h) * 0.95
        sw, sh = int(w * self.current_scale), int(h * self.current_scale)
        self.image_offset_x = (cw - sw) // 2
        self.image_offset_y = (ch - sh) // 2

        self.left_photo = ImageTk.PhotoImage(Image.fromarray(cv2.resize(self.original_img, (sw, sh))))
        self.left_canvas.delete("all")
        self.left_canvas.create_image(cw//2, ch//2, image=self.left_photo)
        self.draw_overlay()

        if self.processed_img is not None:
            rw, rh = self.right_canvas.winfo_width(), self.right_canvas.winfo_height()
            self.right_photo = ImageTk.PhotoImage(Image.fromarray(cv2.resize(self.processed_img, (sw, sh))))
            self.right_canvas.delete("all")
            self.right_canvas.create_image(rw//2, rh//2, image=self.right_photo)

    def draw_overlay(self):
        mode = self.mode_var.get()
        scale = self.current_scale
        ox, oy = self.image_offset_x, self.image_offset_y

        if mode == 0 and self.points:
            for cell in self.triangles:
                pts = []
                for idx in cell:
                    pts.extend([self.points[idx][0]*scale + ox, self.points[idx][1]*scale + oy])
                self.left_canvas.create_polygon(pts, fill="", outline="#00ffcc", width=1)
            for p in self.points:
                px, py = p[0]*scale + ox, p[1]*scale + oy
                self.left_canvas.create_oval(px-4, py-4, px+4, py+4, fill="#ff3366", outline="white")
        
        elif mode == 1 and self.perspective_corners is not None:
            pts = [(p[0]*scale + ox, p[1]*scale + oy) for p in self.perspective_corners]
            self.left_canvas.create_polygon(pts, fill="", outline="#ffaa00", width=2)
            for i, p in enumerate(pts):
                self.left_canvas.create_oval(p[0]-8, p[1]-8, p[0]+8, p[1]+8, fill="#ffaa00")
                self.left_canvas.create_text(p[0], p[1]-15, text=str(i), fill="white")

        elif mode == 2 and self.resize_corners is not None:
            pts = [(p[0]*scale + ox, p[1]*scale + oy) for p in self.resize_corners]
            self.left_canvas.create_polygon(pts, fill="", outline="#00aaff", width=3)
            for p in pts:
                self.left_canvas.create_oval(p[0]-10, p[1]-10, p[0]+10, p[1]+10, fill="#00aaff")

    def init_grid(self):
        if self.original_img is None: return
        h, w = self.original_img.shape[:2]
        n = self.grid_size_var.get()
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
        self.process_warp(self.original_img, cv2.INTER_LINEAR)
        self.update_displays()

    def on_mode_change(self):
        self.current_mode = self.mode_var.get()
        if self.processed_img is not None:
            self.current_base_img = self.processed_img.copy()
        
        if self.current_mode == 0: self.init_grid()
        elif self.current_mode == 1: self.init_perspective()
        elif self.current_mode == 2: self.init_geometry()
        self.update_displays()

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            img = cv2.imread(path)
            self.original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.processed_img = self.original_img.copy()
            self.reset_all()

    def reset_all(self):
        self.current_base_img = None
        if self.original_img is not None:
            self.processed_img = self.original_img.copy()
            self.on_mode_change()
        self.update_displays()

    def init_perspective(self):
        if self.original_img is None: return
        h, w = self.original_img.shape[:2]
        self.perspective_corners = np.array([[50,50], [w-50,50], [w-50,h-50], [50,h-50]], dtype=np.float32)
        self.apply_perspective_transform()

    def init_geometry(self):
        if self.original_img is None: return
        h, w = self.original_img.shape[:2]
        self.resize_corners = np.array([[20,20], [w-20,20], [w-20,h-20], [20,h-20]], dtype=np.float32)
        self.apply_geometric_transform()

    def apply_perspective_transform(self):
        base = self.current_base_img if self.current_base_img is not None else self.original_img
        h, w = base.shape[:2]
        dst = np.float32([[0,0], [w,0], [w,h], [0,h]])
        M = cv2.getPerspectiveTransform(self.perspective_corners, dst)
        self.processed_img = cv2.warpPerspective(base, M, (w, h))

    def apply_geometric_transform(self):
        base = self.current_base_img if self.current_base_img is not None else self.original_img
        h, w = base.shape[:2]
        src = np.float32([[0,0], [w,0], [w,h], [0,h]])
        M = cv2.getPerspectiveTransform(src, self.resize_corners)
        self.processed_img = cv2.warpPerspective(base, M, (w, h), borderValue=(255,255,255))

    def save_image(self):
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path: cv2.imwrite(path, cv2.cvtColor(self.processed_img, cv2.COLOR_RGB2BGR))

    def run_ocr(self):
        if self.processed_img is None:
            messagebox.showwarning("Внимание", "Изображение не загружено!")
            return
        
        # 1. Запрашиваем путь для сохранения (текстового файла)
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Сохранить результат (Текст и Фото)"
        )
        
        if not save_path:
            return

        try:
            # 2. Подготовка и запуск OCR
            gray = cv2.cvtColor(self.processed_img, cv2.COLOR_RGB2GRAY)
            # Небольшое размытие для чистоты OCR
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            text = pytesseract.image_to_string(binary, lang='eng+rus')

            # 3. Сохранение ТЕКСТА
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(text)

            # 4. Сохранение КАРТИНКИ (с тем же именем, но расширением .png)
            img_path = os.path.splitext(save_path)[0] + ".png"
            # Конвертируем RGB -> BGR перед сохранением, чтобы цвета не инвертировались
            final_save_img = cv2.cvtColor(self.processed_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, final_save_img)

            messagebox.showinfo("Успех!", f"Сохранено два файла:\n1. {os.path.basename(save_path)}\n2. {os.path.basename(img_path)}")
        
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить файлы: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRCorrector(root)
    root.mainloop()