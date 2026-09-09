#!/usr/bin/env python3
import os
import shutil
import time
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk


class ImageFilterApp:
    def __init__(self, root, raw_dir, approved_dir, discarded_dir):
        self.root = root
        self.raw_dir = raw_dir
        self.approved_dir = approved_dir
        self.discarded_dir = discarded_dir
        
        self.root.title("PanelSafe - Dataset Image Filter Utility")
        self.root.geometry("900x750")
        self.root.configure(bg="#1c1924")
        
        # Ensure directories exist
        os.makedirs(self.approved_dir, exist_ok=True)
        os.makedirs(self.discarded_dir, exist_ok=True)
        
        # Scan for images
        self.image_paths = []
        if os.path.exists(self.raw_dir):
            for f in os.listdir(self.raw_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.image_paths.append(os.path.join(self.raw_dir, f))
                    
        self.current_index = 0
        self.keep_count = 0
        self.discard_count = 0
        self.skip_count = 0
        
        self.current_img_tk = None
        
        # Build UI layout
        self.create_widgets()
        
        # Key bindings
        self.root.bind("<Left>", lambda e: self.discard_image())
        self.root.bind("<Right>", lambda e: self.keep_image())
        self.root.bind("<Up>", lambda e: self.skip_image())
        self.root.bind("<space>", lambda e: self.skip_image())
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        
        # Load first image
        self.load_current_image()

    def create_widgets(self):
        # Top title/instructions panel
        self.info_frame = tk.Frame(self.root, bg="#1c1924", pady=10)
        self.info_frame.pack(fill=tk.X)
        
        self.title_label = tk.Label(
            self.info_frame, 
            text="Dataset Photo Reviewer", 
            font=("Arial", 16, "bold"), 
            fg="#c084fc", 
            bg="#1c1924"
        )
        self.title_label.pack()
        
        self.instructions_label = tk.Label(
            self.info_frame, 
            text="[Left Arrow] Discard | [Right Arrow] Keep | [Up Arrow / Space] Skip | [Esc] Exit", 
            font=("Arial", 11), 
            fg="#9ca3af", 
            bg="#1c1924"
        )
        self.instructions_label.pack(pady=5)
        
        # Status stats bar
        self.stats_frame = tk.Frame(self.root, bg="#100e16", height=40)
        self.stats_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.stats_label = tk.Label(
            self.stats_frame,
            text="Total: 0 | Kept: 0 | Discarded: 0 | Current: 0/0",
            font=("Arial", 10, "bold"),
            fg="#e5e7eb",
            bg="#100e16",
            pady=8
        )
        self.stats_label.pack()
        
        # Image Display Area (Canvas)
        self.canvas = tk.Canvas(self.root, bg="#100e16", highlightthickness=1, highlightbackground="#3b1d54")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Bottom Buttons
        self.button_frame = tk.Frame(self.root, bg="#1c1924", pady=15)
        self.button_frame.pack(fill=tk.X)
        
        self.discard_btn = tk.Button(
            self.button_frame, 
            text="Discard (<-)", 
            command=self.discard_image,
            bg="#ef4444", 
            fg="#ffffff", 
            activebackground="#dc2626", 
            activeforeground="#ffffff",
            font=("Arial", 12, "bold"), 
            padx=20, 
            pady=10,
            borderwidth=0,
            cursor="hand2"
        )
        self.discard_btn.pack(side=tk.LEFT, padx=50)
        
        self.skip_btn = tk.Button(
            self.button_frame, 
            text="Skip (Up/Space)", 
            command=self.skip_image,
            bg="#4b5563", 
            fg="#ffffff", 
            activebackground="#374151", 
            activeforeground="#ffffff",
            font=("Arial", 12, "bold"), 
            padx=20, 
            pady=10,
            borderwidth=0,
            cursor="hand2"
        )
        self.skip_btn.pack(side=tk.LEFT, expand=True)
        
        self.keep_btn = tk.Button(
            self.button_frame, 
            text="Keep (->)", 
            command=self.keep_image,
            bg="#22c55e", 
            fg="#ffffff", 
            activebackground="#16a34a", 
            activeforeground="#ffffff",
            font=("Arial", 12, "bold"), 
            padx=20, 
            pady=10,
            borderwidth=0,
            cursor="hand2"
        )
        self.keep_btn.pack(side=tk.RIGHT, padx=50)

    def load_current_image(self):
        self.canvas.delete("all")
        
        if not self.image_paths or self.current_index >= len(self.image_paths):
            self.stats_label.config(text=f"Total: {len(self.image_paths)} | Kept: {self.keep_count} | Discarded: {self.discard_count} - Review complete!")
            self.canvas.create_text(
                450, 250, 
                text="All images reviewed!\nYou can close the window.", 
                fill="#22c55e", 
                font=("Arial", 16, "bold"),
                justify=tk.CENTER
            )
            self.discard_btn.config(state=tk.DISABLED)
            self.keep_btn.config(state=tk.DISABLED)
            self.skip_btn.config(state=tk.DISABLED)
            return
            
        img_path = self.image_paths[self.current_index]
        filename = os.path.basename(img_path)
        
        # Update stats
        self.stats_label.config(
            text=f"Total: {len(self.image_paths)} | Kept: {self.keep_count} | Discarded: {self.discard_count} | Current: {self.current_index + 1}/{len(self.image_paths)}"
        )
        
        try:
            # Get canvas current dimensions, or fallback to standard geometries
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            if canvas_w < 100 or canvas_h < 100:
                canvas_w = 860
                canvas_h = 450
                
            img = Image.open(img_path)
            
            # Check dimensions and scale preserving aspect ratio
            img_w, img_h = img.size
            ratio = min(canvas_w / img_w, canvas_h / img_h)
            new_w = max(10, int(img_w * ratio))
            new_h = max(10, int(img_h * ratio))
            
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            self.current_img_tk = ImageTk.PhotoImage(img_resized)
            
            # Draw on canvas center
            cx = canvas_w / 2
            cy = canvas_h / 2
            self.canvas.create_image(cx, cy, anchor=tk.CENTER, image=self.current_img_tk)
            
            # Draw metadata info text at bottom of canvas
            meta_str = f"{filename} ({img_w}x{img_h})"
            self.canvas.create_text(
                10, canvas_h - 15,
                text=meta_str,
                fill="#9ca3af",
                anchor=tk.SW,
                font=("Arial", 10)
            )
        except Exception as e:
            # Handle corrupt images by auto skipping or flagging
            print(f"Error loading image {img_path}: {e}")
            self.canvas.create_text(
                450, 250, 
                text=f"Error loading image:\n{filename}\n{str(e)}\n\nFile will be skipped/flagged.", 
                fill="#ef4444", 
                font=("Arial", 12, "bold"),
                justify=tk.CENTER
            )

    def keep_image(self):
        if self.current_index >= len(self.image_paths):
            return
            
        src_path = self.image_paths[self.current_index]
        filename = os.path.basename(src_path)
        
        # Build a clean unique filename for the approved target
        timestamp = int(time.time())
        ext = os.path.splitext(filename)[1]
        new_filename = f"scraped_approved_{timestamp}_{self.current_index}{ext}"
        dst_path = os.path.join(self.approved_dir, new_filename)
        
        try:
            shutil.move(src_path, dst_path)
            self.keep_count += 1
            print(f"Kept: {filename} -> {new_filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not move file: {e}")
            return
            
        self.current_index += 1
        self.load_current_image()

    def discard_image(self):
        if self.current_index >= len(self.image_paths):
            return
            
        src_path = self.image_paths[self.current_index]
        filename = os.path.basename(src_path)
        dst_path = os.path.join(self.discarded_dir, filename)
        
        try:
            shutil.move(src_path, dst_path)
            self.discard_count += 1
            print(f"Discarded (Moved to safety folder): {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not move file: {e}")
            return
            
        self.current_index += 1
        self.load_current_image()

    def skip_image(self):
        if self.current_index >= len(self.image_paths):
            return
            
        self.skip_count += 1
        self.current_index += 1
        self.load_current_image()

def main():
    # Paths configured relative to project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(project_root, "data", "scraped_raw")
    approved_dir = os.path.join(project_root, "data", "scraped_approved")
    discarded_dir = os.path.join(project_root, "data", "scraped_discarded")
    
    root = tk.Tk()
    _ = ImageFilterApp(root, raw_dir, approved_dir, discarded_dir)
    
    # Run Tkinter mainloop
    root.mainloop()

if __name__ == "__main__":
    main()
