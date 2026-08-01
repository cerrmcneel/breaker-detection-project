"""
breaker_labeler — CSV panel-attribute labeler for the tabular safety-scoring track.

Moved into the main repo from the standalone breaker-labeler project (2026-06-27) so it
lives alongside the model that consumes its output: it writes data/breaker_dataset.csv,
whose columns are read by src/scoring.py (has_rcd, has_rcd_si, has_ovp, panel_age, ...).

NOTE: this is NOT the YOLO bounding-box annotator (that is external LabelImg + the seed
cropper at src/tools/seed_cropper.py). It produces no detection labels.

Run from anywhere:  python -m src.tools.labeler
"""
import csv
import glob
import hashlib
import json
import os
import pathlib
import shutil
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog

from PIL import Image, ImageTk

from src.data_schema import (
    COMMENTS,
    COUNTRY,
    CSV_FIELDNAMES,
    FILENAME,
    HAS_IGA,
    HAS_OVP,
    HAS_RCD,
    HAS_RCD_SI,
    IGA_AMP,
    MCB_VALUES,
    NUM_MCBS,
    NUMBER_RCD,
    PANEL_AGE,
    PHASE_TYPE,
    TIMESTAMP,
)

# Anchor all paths to the project root (src/tools/labeler.py -> parents[2]) so the tool
# works regardless of the current working directory.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = str(PROJECT_ROOT / 'data' / 'images' / 'raw_uploads')
LOG_FILE = str(PROJECT_ROOT / 'data' / 'images' / 'upload_log.json')

CSV_FILE = str(PROJECT_ROOT / 'data' / 'breaker_dataset.csv')
# Column schema is single-sourced in src/data_schema.py so the labeler (writer) and
# src/scoring.py (reader) cannot drift apart.
FIELDNAMES = CSV_FIELDNAMES

# Blurry/unusable images are moved here instead of deleted, so a discard is recoverable.
DISCARD_DIR = str(PROJECT_ROOT / 'data' / 'images' / 'discarded')

def get_country_from_log(filename, log_path=LOG_FILE):
    if not os.path.exists(log_path):
        return 'Unknown'
        
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
            
            # Search for the entry that matches our current filename
            for entry in logs:
                if entry.get('saved_filename') == filename:
                    return entry.get('country', 'Unknown')
                    
    except (json.JSONDecodeError, FileNotFoundError):
        pass
        
    return 'Unknown'

def get_labeled_data():
    labeled_data = {}
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                labeled_data[row['filename']] = row
    return labeled_data

def get_labeled_files():
    return set(get_labeled_data().keys())

def get_all_images(data_dir):
    all_images = []
    # Add different standard image extensions
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        all_images.extend(glob.glob(os.path.join(data_dir, ext)))
    # Deduplicate paths (Windows glob is case-insensitive, so *.jpg and *.JPG will find the same files)
    return list(set(os.path.normpath(p) for p in all_images))

def get_unlabeled_images(data_dir):
    labeled_files = get_labeled_files()
    all_images = get_all_images(data_dir)
        
    unlabeled = [img for img in all_images if os.path.basename(img) not in labeled_files]
    return sorted(unlabeled), len(all_images), len(labeled_files)

def remove_duplicate_images(data_dir):
    all_images = get_all_images(data_dir)
    seen_hashes = set()
    duplicates_removed = 0
    
    for img_path in all_images:
        try:
            with open(img_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                
            if file_hash in seen_hashes:
                os.remove(img_path)
                print(f"[Duplicate Filter] Removed exact duplicate: {os.path.basename(img_path)}")
                duplicates_removed += 1
            else:
                seen_hashes.add(file_hash)
        except Exception as e:
            print(f"[Error] Could not process {img_path} for duplicate check: {e}")
            
    if duplicates_removed > 0:
        print(f"[Duplicate Filter] Successfully removed {duplicates_removed} duplicate files.")

def save_annotation(data, filename):
    file_exists = os.path.exists(CSV_FILE)
    
    # Check if a country was provided explicitly by the user, otherwise try JSON log
    country = data.get('country', '')
    if not country or country == 'Unknown':
        country = get_country_from_log(filename)
        
        # Optional fallback exactly as you had it before, just in case! 
        if country == 'Unknown':
            parts = filename.split('_')
            country = parts[0] if len(parts) > 1 else 'Unknown'
    
    mcb_values = data.get('mcb_values', '').strip()
    if mcb_values:
        # Number of MCBs by splitting commas and filtering out empty strings
        num_mcbs = len([x for x in mcb_values.split(',') if x.strip()])
    else:
        num_mcbs = 0
        
    row = {
        FILENAME: filename,
        TIMESTAMP: datetime.now().isoformat(timespec='seconds'),
        COUNTRY: country,
        PANEL_AGE: data.get(PANEL_AGE, ''),
        PHASE_TYPE: data.get(PHASE_TYPE, ''),
        HAS_RCD: data.get(HAS_RCD, ''),
        HAS_RCD_SI: data.get(HAS_RCD_SI, ''),
        HAS_IGA: data.get(HAS_IGA, ''),
        IGA_AMP: data.get(IGA_AMP, ''),
        HAS_OVP: data.get(HAS_OVP, ''),
        MCB_VALUES: mcb_values,
        NUM_MCBS: num_mcbs,
        COMMENTS: data.get(COMMENTS, ''),
        NUMBER_RCD: data.get(NUMBER_RCD, '')
    }
    
    rows = []
    if file_exists:
        with open(CSV_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
    found = False
    for i, r in enumerate(rows):
        if r['filename'] == filename:
            rows[i] = row
            found = True
            break
            
    if not found:
        rows.append(row)
    
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in rows:
            filtered_r = {k: v for k, v in r.items() if k in FIELDNAMES}
            writer.writerow(filtered_r)

def remove_csv_row(filename):
    """Drop any existing CSV row for filename (used when discarding an already-labeled image)."""
    if not os.path.exists(CSV_FILE):
        return
    with open(CSV_FILE, mode='r', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['filename'] != filename]
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: v for k, v in r.items() if k in FIELDNAMES})

def discard_image(data_dir, img_path, filename):
    """
    Move a blurry/unusable image (and its sidecar YOLO box-label .txt, if any) out of
    data_dir into DISCARD_DIR, and drop any existing CSV row for it. Moves rather than
    deletes so the decision is recoverable.
    """
    os.makedirs(DISCARD_DIR, exist_ok=True)
    stem = os.path.splitext(filename)[0]
    sidecar = os.path.join(data_dir, stem + '.txt')
    moved = []
    for src in [img_path, sidecar]:
        if not os.path.exists(src):
            continue
        dst = os.path.join(DISCARD_DIR, os.path.basename(src))
        if os.path.exists(dst):
            ts = int(datetime.now().timestamp())
            base, ext = os.path.splitext(os.path.basename(src))
            dst = os.path.join(DISCARD_DIR, f"{base}_{ts}{ext}")
        shutil.move(src, dst)
        moved.append(os.path.basename(src))
    remove_csv_row(filename)
    print(f"[DISCARDED] Moved {', '.join(moved)} -> {DISCARD_DIR}")

class LabelerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Breaker Labeler Image Viewer")
        self.root.geometry("800x800")
        
        # Bring window to front so dialog is visible
        self.root.attributes('-topmost', True)
        
        self.data_dir = filedialog.askdirectory(
            initialdir=DATA_DIR,
            title="Select Image Folder"
        )
        self.root.attributes('-topmost', False)
        
        if not self.data_dir:
            print("No folder selected. Exiting...")
            os._exit(0)
            
        print("\nScanning selected folder for duplicate content...")
        remove_duplicate_images(self.data_dir)
            
        # Handle the user closing the window via the 'X' button
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.canvas = tk.Canvas(self.root, bg="gray")
        self.canvas.pack(expand=True, fill="both")
        
        self.unlabeled_images, self.total_images, self.total_labeled = get_unlabeled_images(self.data_dir)
        self.photo = None
        self.original_image = None
        self.zoom_factor = 1.0
        self.pan_x = 400
        self.pan_y = 400
        self.start_x = 0
        self.start_y = 0
        self.image_item = None
        
        # Configure mouse scroll to zoom
        self.canvas.bind("<MouseWheel>", self.on_mouse_zoom)
        self.canvas.bind("<Button-4>", self.on_mouse_zoom)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_zoom)  # Linux scroll down
        
        # Configure mouse drag to pan
        self.canvas.bind("<ButtonPress-1>", self.on_pan_start)
        self.canvas.bind("<B1-Motion>", self.on_pan_motion)
        
        # Configure key to rotate
        self.root.bind("r", self.on_rotate)
        self.root.bind("R", self.on_rotate)
        
        # We start the CLI loop in a daemon thread so it runs alongside the UI
        self.cli_thread = threading.Thread(target=self.cli_loop, daemon=True)
        
    def on_close(self):
        self.print_summary()
        print("\n[GUI Closed] Exiting safely...")
        os._exit(0)
        
    def on_mouse_zoom(self, event):
        if not self.original_image:
            return
            
        # Respond to scroll
        if event.num == 5 or event.delta < 0:  # scroll down / out
            self.zoom_factor = max(0.2, self.zoom_factor * 0.9)
        if event.num == 4 or event.delta > 0:  # scroll up / in
            self.zoom_factor = min(5.0, self.zoom_factor * 1.1)
            
        self.update_image_display()

    def on_pan_start(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def on_pan_motion(self, event):
        dx = event.x - self.start_x
        dy = event.y - self.start_y
        self.pan_x += dx
        self.pan_y += dy
        self.start_x = event.x
        self.start_y = event.y
        self.update_image_display()

    def on_rotate(self, event):
        if self.original_image:
            self.original_image = self.original_image.rotate(-90, expand=True)
            self.update_image_display()

    def show_image(self, path):
        try:
            self.original_image = Image.open(path)
            self.zoom_factor = 1.0 # reset zoom
            self.pan_x = self.root.winfo_width() // 2 if self.root.winfo_width() > 1 else 400
            self.pan_y = self.root.winfo_height() // 2 if self.root.winfo_height() > 1 else 400
            self.update_image_display()
        except Exception as e:
            print(f"\n[Error] Could not load image {path}: {e}")
            
    def update_image_display(self):
        if not self.original_image: return
        
        # Calculate new sizes based on zoom factor
        width, height = self.original_image.size
        new_size = (max(1, int(width * self.zoom_factor)), max(1, int(height * self.zoom_factor)))
        
        img_resized = self.original_image.resize(new_size, Image.Resampling.LANCZOS)
            
        self.photo = ImageTk.PhotoImage(img_resized)
        if self.image_item is None:
            self.image_item = self.canvas.create_image(self.pan_x, self.pan_y, image=self.photo, anchor="center")
        else:
            self.canvas.itemconfig(self.image_item, image=self.photo)
            self.canvas.coords(self.image_item, self.pan_x, self.pan_y)
            
    def print_summary(self):
        print(f"\n{'='*30}")
        print("SESSION SUMMARY")
        print(f"{'='*30}")
        labeled = len(get_labeled_files())
        total = max(1, len(get_all_images(self.data_dir)))
        print(f"Total Photos Submitted: {total}")
        print(f"Total Labeled: {labeled}")
        print(f"Pending to Label: {total - labeled}")
        print(f"Progress to Goal (100): {min(100.0, (labeled / 100) * 100):.1f}%")
        print(f"{'='*30}\n")
            
    def cli_loop(self):
        self.print_summary()
        
        mode_str = input("\nDo you want to (N)ew images or (R)eview existing? [Default: N]: ").strip().lower()
        self.labeled_data = get_labeled_data()
        self.is_review_mode = (mode_str == 'r')
        
        if self.is_review_mode:
            all_images = get_all_images(self.data_dir)
            all_basenames = {os.path.basename(p): p for p in all_images}
            
            review_images = []
            for f in self.labeled_data.keys():
                if f in all_basenames:
                    review_images.append(all_basenames[f])
            
            self.unlabeled_images = sorted(review_images)
            if not self.unlabeled_images:
                print("No labeled images found in directory to review.")
                self.root.after(0, self.root.destroy)
                return
        else:
            if not self.unlabeled_images:
                print(f"No unlabeled images found in '{self.data_dir}'.")
                print("Please add images to continue.")
                self.root.after(0, self.root.destroy)
                return
        
        img_index = 0
        while img_index < len(self.unlabeled_images):
            img_path = self.unlabeled_images[img_index]
            filename = os.path.basename(img_path)
            
            # Request the main thread to update the image
            self.root.after(0, self.show_image, img_path)
            
            print(f"\n{'='*50}")
            print(f"Current Image {img_index + 1}/{len(self.unlabeled_images)}: {filename}")
            print(f"{'='*50}")
            
            prompt_msg = "Press 's' to skip, 'd' to DISCARD (blurry/unusable), 'p' for previous image, 'q' to save and quit"
            if self.is_review_mode:
                prompt_msg += ", 'e' to force edit all, or ENTER to label missing: "
            else:
                prompt_msg += ", or ENTER to label: "

            action = input(prompt_msg).strip().lower()

            force_edit = False
            if action == 'q':
                print("Quitting safely...")
                self.root.after(0, self.root.destroy)
                return
            elif action == 'p':
                if img_index > 0:
                    img_index -= 1
                else:
                    print("Already at the first image.")
                continue
            elif action == 's':
                print(f"Skipping {filename}...")
                img_index += 1
                continue
            elif action == 'd':
                discard_image(self.data_dir, img_path, filename)
                self.labeled_data.pop(filename, None)
                # Remove from the in-memory queue (list shrinks; img_index now points at
                # what was the next image, so it is intentionally NOT incremented here).
                del self.unlabeled_images[img_index]
                continue
            elif action == 'e' and self.is_review_mode:
                force_edit = True
                
            try:
                # Try to get the dynamic log path matching the selected data_dir
                log_path = os.path.join(os.path.dirname(self.data_dir), 'upload_log.json')
                if not os.path.exists(log_path):
                    log_path = LOG_FILE
                    
                guessed_country = get_country_from_log(filename, log_path)
                if guessed_country == 'Unknown':
                    parts = filename.split('_')
                    guessed_country = parts[0] if len(parts) > 1 else 'Unknown'

                # Data entry
                print("\n--- Enter Data (Leave blank if unknown, type '<' to go back one column) ---")
                
                existing_row = self.labeled_data.get(filename, {})
                answers = existing_row.copy()
                
                all_fields = [COUNTRY, PHASE_TYPE, HAS_RCD, HAS_RCD_SI, NUMBER_RCD, HAS_IGA, IGA_AMP, HAS_OVP, MCB_VALUES, COMMENTS, PANEL_AGE]
                
                if self.is_review_mode and not force_edit:
                    fields = [f for f in all_fields if not existing_row.get(f)]
                    if not fields:
                        print("All fields are already filled for this image. Skipping prompt...")
                else:
                    fields = all_fields
                    
                field_idx = 0
                
                while field_idx < len(fields):
                    field = fields[field_idx]
                    val = ""
                    def_val = existing_row.get(field, "")
                    
                    if field == "country":
                        print(f"Detected Country: {guessed_country}")
                        if not def_val: def_val = guessed_country
                        c_in = input(f"Select Country (e.g. ES, PT, FR) [Press ENTER for {def_val}]: ").strip()
                        if c_in == '<':
                            print("Already at the first column.")
                            continue
                        val = c_in if c_in else def_val
                            
                    elif field == "phase_type":
                        prompt = f"Phase Type - Single (1) or Three Phase (3) [{def_val}]: " if def_val else "Phase Type - Single (1) or Three Phase (3): "
                        val_in = input(prompt).strip()
                        if val_in == '<': val = '<'
                        else: val = val_in if val_in else def_val
                    elif field == "has_rcd":
                        prompt = f"RCD Standard - has_rcd (0/1) [{def_val}]: " if def_val else "RCD Standard - has_rcd (0/1): "
                        val_in = input(prompt).strip()
                        if val_in == '<': val = '<'
                        else: val = val_in if val_in else def_val
                    elif field == "has_rcd_si":
                        prompt = f"RCD Super-Immunized - has_rcd_si (0/1) [{def_val}]: " if def_val else "RCD Super-Immunized - has_rcd_si (0/1): "
                        val_in = input(prompt).strip()
                        if val_in == '<': val = '<'
                        else: val = val_in if val_in else def_val
                    elif field == "number_rcd":
                        prompt = f"Number of RCDs - number_rcd (integer) [{def_val}]: " if def_val else "Number of RCDs - number_rcd (integer): "
                        val_in = input(prompt).strip()
                        if val_in == '<': val = '<'
                        else: val = val_in if val_in else def_val
                    elif field == "has_iga":
                        prompt = f"IGA Check - has_iga (0/1) [{def_val}]: " if def_val else "IGA Check - has_iga (0/1): "
                        val_in = input(prompt).strip()
                        if val_in == '<': val = '<'
                        else: val = val_in if val_in else def_val
                    elif field == "iga_amp":
                        prompt = f"IGA Check - iga_amp (integer, e.g., 25, 40) [{def_val}]: " if def_val else "IGA Check - iga_amp (integer, e.g., 25, 40): "
                        val_in = input(prompt).strip()
                        if val_in == '<': val = '<'
                        else: val = val_in if val_in else def_val
                    elif field == "has_ovp":
                        prompt = f"Protection - has_ovp (surge protection, 0/1) [{def_val}]: " if def_val else "Protection - has_ovp (surge protection, 0/1): "
                        val_in = input(prompt).strip()
                        if val_in == '<': val = '<'
                        else: val = val_in if val_in else def_val
                    elif field == "mcb_values":
                        prompt = f"Breaker Array - mcb_values (amperages separated by commas, e.g., '10,16,16,20') [{def_val}]: " if def_val else "Breaker Array - mcb_values (amperages separated by commas, e.g., '10,16,16,20'): "
                        mcb_raw = input(prompt).strip()
                        if mcb_raw == '<':
                            val = '<'
                        else:
                            val_raw = mcb_raw if mcb_raw else def_val
                            try:
                                nums = [int(x.strip()) for x in val_raw.split(',') if x.strip()]
                                nums.sort()
                                val = ','.join(map(str, nums))
                            except ValueError:
                                print("Warning: Non-integer MCB values detected.")
                                val = val_raw
                    elif field == "comments":
                        prompt = f"Expert Note - comments [{def_val}]: " if def_val else "Expert Note - comments: "
                        val_in = input(prompt).strip()
                        if val_in == '<': val = '<'
                        else: val = val_in if val_in else def_val
                    elif field == "panel_age":
                        print("\nOptions for panel_age:")
                        print("1: < 5 years")
                        print("2: 5-10 years")
                        print("3: 10-15 years")
                        print("4: 15 years")
                        print("5: > 20 years")
                        prompt = f"Select panel age (1-5) [Current: {def_val}]: " if def_val else "Select panel age (1-5) [Leave blank if unknown]: "
                        age_idx = input(prompt).strip()
                        if age_idx == '<':
                            val = '<'
                        elif age_idx == '':
                            val = def_val
                        else:
                            if age_idx == '1': val = "< 5 years"
                            elif age_idx == '2': val = "5-10 years"
                            elif age_idx == '3': val = "10-15 years"
                            elif age_idx == '4': val = "15 years"
                            elif age_idx == '5': val = "> 20 years"
                            else: val = def_val
                    
                    if val == '<':
                        field_idx -= 1
                        continue
                        
                    answers[field] = val
                    field_idx += 1
                
                save_annotation(answers, filename)
                self.labeled_data[filename] = answers
                print(f"[Success] Saved data for {filename}.")
                img_index += 1
                
            except Exception as e:
                print(f"[Error] Failed to save data: {e}")
                
        print("\nAll available images have been processed!")
        self.root.after(0, self.root.destroy)

    def run(self):
        self.cli_thread.start()
        # Start the Tkinter event loop on the main thread
        self.root.mainloop()

if __name__ == '__main__':
    app = LabelerApp()
    app.run()
