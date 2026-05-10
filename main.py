import customtkinter as ctk
from PIL import Image, ImageDraw, ImageOps
import tkinter.filedialog as fd
import tkinter.messagebox as mb
import numpy as np
import threading
import traceback
from Processor import ImageProcessor

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --- Modern Tema Ayarları ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Pencere Konfigürasyonu ---
        self.title("VisionCraft Pro | Görüntü İşleme Stüdyosu")
        self.geometry("1400x850")
        self.minsize(1000, 700)
        
        # Renk Paleti
        self.colors = {
            "bg": "#0f172a",
            "sidebar": "#1e293b",
            "accent": "#38bdf8",
            "accent_hover": "#0ea5e9",
            "danger": "#ef4444",
            "success": "#10b981",
            "text": "#f8fafc",
            "card": "#334155"
        }

        self.configure(fg_color=self.colors["bg"])

        self.original_image_matrix = None
        self.current_image_matrix = None
        self.second_image_matrix = None

        self._ctk_img_orig = None
        self._ctk_img_mod = None
        self._progress_reset_job = None

        self.setup_ui()

    def setup_ui(self):
        # --- Üst Panel (Header) ---
        self.header_frame = ctk.CTkFrame(self, height=82, corner_radius=0, fg_color=self.colors["sidebar"])
        self.header_frame.pack(side="top", fill="x")
        self.header_frame.pack_propagate(False)

        # Hızlı Aksiyon Butonları (Sağ Üst)
        self.top_btn_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.top_btn_frame.pack(side="right", padx=24, pady=14)

        self.btn_load = self.create_top_button("📂 Resim Aç", self.load_image, self.colors["accent"])
        self.btn_load2 = self.create_top_button("➕ 2. Resim", self.load_second_image, "#818cf8")
        self.btn_clear2 = self.create_top_button("2. Resmi Sil", self.clear_second_image, self.colors["danger"])
        self.btn_save = self.create_top_button("💾 Kaydet", self.save_current, self.colors["success"])

        # --- Ana İçerik ---
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Sol Sidebar (Kontroller)
        self.sidebar = ctk.CTkFrame(self.content_frame, width=320, corner_radius=15, fg_color=self.colors["sidebar"])
        self.sidebar.pack(side="left", fill="y", padx=(0, 20))
        self.sidebar.pack_propagate(False)

        ctk.CTkLabel(
            self.sidebar, text="KONTROL PANELİ", 
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#94a3b8"
        ).pack(pady=(20, 10))

        # Tabview - Filtre Grupları
        self.tabs = ctk.CTkTabview(
            self.sidebar, 
            segmented_button_selected_color=self.colors["accent"],
            segmented_button_unselected_hover_color="#334155",
            fg_color="transparent"
        )
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_basic = self.tabs.add("Temel")
        self.tab_color = self.tabs.add("Renk")
        self.tab_filter = self.tabs.add("Filtre")
        self.tab_adv = self.tabs.add("İleri")

        self.setup_filters()

        # Alt işlem butonları
        self.bottom_actions = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.bottom_actions.pack(side="bottom", fill="x", padx=20, pady=(5, 20))

        self.status_label = ctk.CTkLabel(self.bottom_actions, text="Hazır", font=ctk.CTkFont(size=11), text_color="#64748b")
        self.status_label.pack(fill="x", pady=(0, 5))

        self.progress = ctk.CTkProgressBar(self.bottom_actions, mode="determinate", height=8, progress_color=self.colors["accent"])
        self.progress.pack(fill="x", pady=(0, 10))
        self.progress.set(0)

        self.btn_apply = ctk.CTkButton(
            self.bottom_actions, text="ALGORİTMAYI ÇALIŞTIR",
            height=45, corner_radius=8,
            font=ctk.CTkFont(weight="bold"),
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"],
            command=self.apply_filter
        )
        self.btn_apply.pack(fill="x", pady=(0, 8))

        self.btn_reset = ctk.CTkButton(
            self.bottom_actions, text="SIFIRLA",
            height=45, corner_radius=8,
            font=ctk.CTkFont(weight="bold"),
            fg_color=self.colors["danger"],
            hover_color="#b91c1c",
            command=self.reset_image
        )
        self.btn_reset.pack(fill="x")

        # Sağ Panel (Görüntüleme)
        self.view_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.view_frame.pack(side="right", fill="both", expand=True)

        # Başlıklar
        self.title_frame = ctk.CTkFrame(self.view_frame, fg_color="transparent", height=40)
        self.title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(self.title_frame, text="ORİJİNAL", font=ctk.CTkFont(weight="bold"), text_color=self.colors["success"]).place(relx=0.25, rely=0.5, anchor="center")
        ctk.CTkLabel(self.title_frame, text="İŞLENMİŞ", font=ctk.CTkFont(weight="bold"), text_color=self.colors["accent"]).place(relx=0.75, rely=0.5, anchor="center")

        # Görüntü Panelleri
        self.image_area = ctk.CTkFrame(self.view_frame, fg_color="#1e293b", corner_radius=20)
        self.image_area.pack(fill="both", expand=True)

        self.panel_orig = ctk.CTkLabel(self.image_area, text="Resim Bekleniyor...", text_color="#475569")
        self.panel_orig.place(relx=0.25, rely=0.5, anchor="center")

        self.panel_mod = ctk.CTkLabel(self.image_area, text="İşlem Bekleniyor...", text_color="#475569")
        self.panel_mod.place(relx=0.75, rely=0.5, anchor="center")

        # Orta Çizgi (Separator)
        self.sep = ctk.CTkFrame(self.image_area, width=2, fg_color="#334155")
        self.sep.place(relx=0.5, rely=0.1, relheight=0.8, anchor="center")

    def create_top_button(self, text, command, color):
        btn = ctk.CTkButton(
            self.top_btn_frame, text=text, command=command,
            fg_color="transparent", border_width=1, border_color=color,
            hover_color=color, text_color=color, height=40, width=110
        )
        btn.pack(side="left", padx=6)
        return btn

    def setup_filters(self):
        # Gruplanmış Filtre Listeleri
        filters = {
            "basic": [
                "1. Gri Seviye", "2. Thresholding", "9. Görüntü Döndürme", 
                "10. Zoom (1.5x)", "29. Görüntü Kırpma"
            ],
            "color": [
                "3. Kontrast Ayarı", "4. Histogram Germe", "5. Histogram Eşitleme",
                "6. Histogram Görüntüle", "7. RGB → HSV", "8. RGB → YCbCr"
            ],
            "filter": [
                "11. Salt & Pepper", "12. Mean Filtre", "13. Median Filtre",
                "14. Motion Blur", "15. Sobel Edge", "16. Canny Edge"
            ],
            "adv": [
                "17. Morph Erosion", "18. Morph Dilation", "19. Morph Opening", "20. Morph Closing",
                "21. Aritmetik: Toplama", "22. Aritmetik: Çıkarma", "23. Aritmetik: Çarpma",
                "24. Aritmetik: AND", "25. Aritmetik: OR", "26. Aritmetik: XOR",
                "27. Adaptif Eşikleme", "28. Çift Eşikleme", "30. PLAKA OKUMA (LPR)"
            ]
        }

        self.filter_vars = {}
        for tab_key, items in filters.items():
            parent = getattr(self, f"tab_{tab_key}")
            scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent", height=400)
            scroll.pack(fill="both", expand=True)
            
            var = ctk.StringVar(value=items[0])
            self.filter_vars[tab_key] = var
            
            for item in items:
                rb = ctk.CTkRadioButton(
                    scroll, text=item, variable=var, value=item,
                    hover_color=self.colors["accent"], border_color="#475569",
                    fg_color=self.colors["accent"]
                )
                rb.pack(pady=10, padx=10, anchor="w")

    # --- Backend Entegrasyonu (KODLARA DOKUNULMADI) ---
    def set_status(self, msg):
        self.status_label.configure(text=msg)
        self.update_idletasks()

    def _reset_progress_idle(self):
        self.progress.set(0)
        self.progress.configure(progress_color=self.colors["accent"], mode="determinate")
        self.status_label.configure(text_color="#64748b")
        self._progress_reset_job = None

    def _read_image(self, filepath):
        """PIL kullanarak görüntüyü numpy array olarak okur."""
        try:
            pil_img = Image.open(filepath)
            pil_img = ImageOps.exif_transpose(pil_img) # EXIF bilgilerine göre resmi düzelt (Yan gelme sorunu)
            pil_img = pil_img.convert("RGB")
            return np.array(pil_img)
        except Exception as e:
            return None

    def load_image(self):
        filepath = fd.askopenfilename(filetypes=[("Görüntüler", "*.png *.jpg *.jpeg *.bmp *.tif")])
        if filepath:
            try:
                img_array = self._read_image(filepath)
                if img_array is not None:
                    self.original_image_matrix = img_array
                    self.current_image_matrix = self.original_image_matrix.copy()
                    self.display_images()
                    self.set_status(f"Yüklendi: {img_array.shape[1]}x{img_array.shape[0]}")
                else:
                    raise Exception("Resim okunamadı")
            except: mb.showerror("Hata", "Resim yüklenemedi!")

    def load_second_image(self):
        filepath = fd.askopenfilename(filetypes=[("Görüntüler", "*.png *.jpg *.jpeg *.bmp *.tif")])
        if filepath:
            img_array = self._read_image(filepath)
            if img_array is not None:
                self.second_image_matrix = img_array
                self.btn_load2.configure(text=f"2. Resim: {img_array.shape[1]}x{img_array.shape[0]}", text_color=self.colors["success"])
                self.set_status("2. resim aritmetik islemler icin hazir.")
            else:
                mb.showerror("Hata", "2. resim okunamadı!")

    def clear_second_image(self):
        self.second_image_matrix = None
        self.btn_load2.configure(text="➕ 2. Resim", text_color="#818cf8")
        self.set_status("2. resim kaldirildi.")

    def save_current(self):
        if self.current_image_matrix is not None:
            path = fd.asksaveasfilename(defaultextension=".png")
            if path: Image.fromarray(self.current_image_matrix).save(path)

    def reset_image(self):
        if self.original_image_matrix is not None:
            self.current_image_matrix = self.original_image_matrix.copy()
            self.display_images()

    def display_images(self):
        max_size = (500, 500)
        for mat, label, attr in [(self.original_image_matrix, self.panel_orig, "_ctk_img_orig"), 
                                (self.current_image_matrix, self.panel_mod, "_ctk_img_mod")]:
            if mat is not None:
                img = Image.fromarray(mat.astype(np.uint8))
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                ctk_img = ctk.CTkImage(img, size=img.size)
                setattr(self, attr, ctk_img)
                label.configure(image=ctk_img, text="")

    def apply_filter(self):
        if self.current_image_matrix is None:
            mb.showwarning("Uyari", "Once bir resim yukleyin.")
            return
        tab = self.tabs.get()
        choice = self.filter_vars[{"Temel":"basic","Renk":"color","Filtre":"filter","İleri":"adv"}[tab]].get()
        if "Aritmetik:" in choice and self.second_image_matrix is None:
            mb.showwarning("Uyari", "Bu islem icin once 2. resmi yuklemelisiniz.")
            self.set_status("2. resim bekleniyor.")
            return
        if self._progress_reset_job is not None:
            self.after_cancel(self._progress_reset_job)
            self._progress_reset_job = None
        self.set_status(f"İşleniyor: {choice}...")
        self.status_label.configure(text_color=self.colors["accent"])
        self.progress.configure(mode="indeterminate", progress_color=self.colors["accent"])
        self.progress.start()
        self.btn_apply.configure(state="disabled")
        threading.Thread(target=self._run_filter, args=(choice,), daemon=True).start()

    def _run_filter(self, choice):
        try:
            mat = self.current_image_matrix.copy()
            result = None
            
            # --- ALGORİTMA EŞLEŞTİRMELERİ (Birebir Aynı) ---
            if "1. Gri Seviye" in choice: result = ImageProcessor.turn_gray(mat)
            elif "2. Thresholding" in choice: result = ImageProcessor.turn_binary(mat, threshold=127)
            elif "3. Kontrast Ayarı" in choice: result = ImageProcessor.adjust_contrast_manual(mat, factor=0.5)
            elif "4. Histogram Germe" in choice: result = ImageProcessor.stretch_histogram_manual(mat)
            elif "5. Histogram Eşitleme" in choice: result = ImageProcessor.histogram_equalization_manual(mat)
            elif "6. Histogram Görüntüle" in choice:
                hist = ImageProcessor.get_histogram(mat)
                self.after(0, lambda: self._show_histogram(hist))
                self.after(0, lambda: self._finish_filter("Histogram ✓"))
                return
            elif "7. RGB → HSV" in choice: result = ImageProcessor.rgb_to_hsv_manual(mat)
            elif "8. RGB → YCbCr" in choice: result = ImageProcessor.rgb_to_ycbcr_manual(mat)
            elif "9. Görüntü Döndürme" in choice: result = ImageProcessor.rotation_image(mat)
            elif "10. Zoom" in choice:
                h, w = mat.shape[:2]
                nh, nw = int(h / 1.5), int(w / 1.5)
                y1, x1 = (h - nh) // 2, (w - nw) // 2
                cropped = mat[y1:y1+nh, x1:x1+nw]
                result = ImageProcessor.resize_manual(cropped, 1.5)
            elif "11. Salt & Pepper" in choice: result = ImageProcessor.add_salt_pepper_noise_manual(mat)
            elif "12. Mean Filtre" in choice: result = ImageProcessor.mean_filter_manual(mat)
            elif "13. Median Filtre" in choice: result = ImageProcessor.median_filter_manual(mat)
            elif "14. Motion Blur" in choice: result = ImageProcessor.motion_blur_manual(mat)
            elif "15. Sobel Edge" in choice: result = ImageProcessor.sobel_edge_manual(mat)
            elif "16. Canny Edge" in choice: result = ImageProcessor.canny_edge_manual(mat)
            elif "17. Morph Erosion" in choice: result = ImageProcessor.turn_erode(mat)
            elif "18. Morph Dilation" in choice: result = ImageProcessor.turn_dilate(mat)
            elif "19. Morph Opening" in choice: result = ImageProcessor.turn_opening(mat)
            elif "20. Morph Closing" in choice: result = ImageProcessor.turn_closing(mat)
            elif "21. Aritmetik: Toplama" in choice: result = ImageProcessor.add_images_manual(mat, self.second_image_matrix)
            elif "22. Aritmetik: Çıkarma" in choice: result = ImageProcessor.subtract_images_manual(mat, self.second_image_matrix)
            elif "23. Aritmetik: Çarpma" in choice: result = ImageProcessor.multiply_images_manual(mat, self.second_image_matrix)
            elif "24. Aritmetik: AND" in choice: result = ImageProcessor.bitwise_and_manual(mat, self.second_image_matrix)
            elif "25. Aritmetik: OR" in choice: result = ImageProcessor.bitwise_or_manual(mat, self.second_image_matrix)
            elif "26. Aritmetik: XOR" in choice: result = ImageProcessor.bitwise_xor_manual(mat, self.second_image_matrix)
            elif "27. Adaptif Eşikleme" in choice: result = ImageProcessor.adaptive_threshold_manual(mat)
            elif "28. Çift Eşikleme" in choice: result = ImageProcessor.double_threshold_manual(mat)
            elif "29. Görüntü Kırpma" in choice:
                h, w = mat.shape[:2]
                result = ImageProcessor.crop_image(mat, w//4, h//4, w//2, h//2)
            elif "30. PLAKA OKUMA" in choice:
                self.after(0, lambda: mb.showinfo("Bilgi", "Plaka okuma henuz uygulanmadi."))
                self.after(0, lambda: self._finish_filter("Islem uygulanmadi."))
                return

            if result is not None:
                self.current_image_matrix = result.astype(np.uint8)
            self.after(0, lambda: self._finish_filter("Başarıyla uygulandı ✓"))
        except Exception as e:
            self.after(0, lambda: mb.showerror("Hata", str(e)))
            self.after(0, lambda: self._finish_filter("Hata!"))

    def _finish_filter(self, msg):
        self.progress.stop()
        self.progress.configure(mode="determinate")
        if "Hata" in msg:
            self.progress.configure(progress_color=self.colors["danger"])
            self.status_label.configure(text_color=self.colors["danger"])
        elif "uygulanmadi" in msg.lower():
            self.progress.configure(progress_color="#f59e0b")
            self.status_label.configure(text_color="#f59e0b")
        else:
            self.progress.configure(progress_color=self.colors["success"])
            self.status_label.configure(text_color=self.colors["success"])
        self.progress.set(1)
        self.btn_apply.configure(state="normal")
        self.display_images()
        self.set_status(msg)
        self._progress_reset_job = self.after(1200, self._reset_progress_idle)

    def _show_histogram(self, hist):
        plt.figure("Histogram", figsize=(6,4))
        plt.clf()
        plt.bar(range(256), hist, color='#38bdf8', width=1)
        plt.title("Piksel Yoğunluğu")
        plt.show(block=False)

if __name__ == "__main__":
    app = App()
    app.mainloop()
