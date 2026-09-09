import hashlib
import os

from PIL import Image


def get_file_md5(filepath):
    """Calculate the MD5 hash of the file contents."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_perceptual_hash(image_path):
    """Generate an 8x8 average hash (aHash) for the image."""
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale and resize to 8x8
            img = img.convert('L').resize((8, 8), Image.Resampling.BILINEAR)
            pixels = list(img.getdata())
            avg = sum(pixels) / 64
            # Generate 64-bit hash representation
            return "".join(["1" if p > avg else "0" for p in pixels])
    except Exception:
        # Silently fail for corrupt images (will be handled during review)
        return None

def hamming_distance(hash1, hash2):
    """Calculate the Hamming distance between two binary hash strings."""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def deduplicate_directory(directory_path, threshold=2):
    """Scans a single directory for internal duplicates and removes them."""
    print(f"Scanning directory for internal duplicates: {directory_path}")
    if not os.path.exists(directory_path):
        return
        
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
             
    md5_map = {}
    phash_map = {}
    
    discard_dir = os.path.join(os.path.dirname(directory_path), "scraped_discarded")
    os.makedirs(discard_dir, exist_ok=True)
    
    removed_count = 0
    
    for filepath in files:
        # 1. MD5 Check
        try:
            md5_val = get_file_md5(filepath)
            if md5_val in md5_map:
                print(f"Internal Byte Duplicate: {os.path.basename(filepath)} matches {os.path.basename(md5_map[md5_val])}")
                os.rename(filepath, os.path.join(discard_dir, os.path.basename(filepath)))
                removed_count += 1
                continue
            md5_map[md5_val] = filepath
        except Exception:
            continue
            
        # 2. Perceptual Check
        phash_val = get_perceptual_hash(filepath)
        if not phash_val:
            continue
            
        is_near_dup = False
        for existing_hash, existing_path in phash_map.items():
            if hamming_distance(phash_val, existing_hash) <= threshold:
                print(f"Internal Perceptual Duplicate: {os.path.basename(filepath)} matches {os.path.basename(existing_path)}")
                if os.path.exists(filepath):
                    os.rename(filepath, os.path.join(discard_dir, os.path.basename(filepath)))
                    removed_count += 1
                is_near_dup = True
                break
                
        if not is_near_dup:
            phash_map[phash_val] = filepath
            
    print(f"Internal deduplication complete. Removed {removed_count} duplicates.")

def deduplicate_cross_directories(source_dir, reference_dir, threshold=2):
    """Removes images from source_dir that are duplicates of images in reference_dir."""
    print(f"Deduplicating {source_dir} against reference {reference_dir}...")
    if not os.path.exists(source_dir) or not os.path.exists(reference_dir):
        return
        
    ref_files = [os.path.join(reference_dir, f) for f in os.listdir(reference_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                 
    src_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                 
    # Build reference maps
    ref_md5s = set()
    ref_phashes = {} # hash -> path
    
    for filepath in ref_files:
        try:
            ref_md5s.add(get_file_md5(filepath))
        except Exception:
            pass
        phash = get_perceptual_hash(filepath)
        if phash:
            ref_phashes[phash] = filepath
            
    discard_dir = os.path.join(os.path.dirname(source_dir), "scraped_discarded")
    os.makedirs(discard_dir, exist_ok=True)
    
    removed_count = 0
    
    for filepath in src_files:
        # 1. Check MD5 against reference
        try:
            md5_val = get_file_md5(filepath)
            if md5_val in ref_md5s:
                print(f"Cross Byte Duplicate: {os.path.basename(filepath)} already exists in approved/reference.")
                os.rename(filepath, os.path.join(discard_dir, os.path.basename(filepath)))
                removed_count += 1
                continue
        except Exception:
            pass
            
        # 2. Check Perceptual Hash against reference
        phash_val = get_perceptual_hash(filepath)
        if not phash_val:
            continue
            
        for ref_hash, ref_path in ref_phashes.items():
            if hamming_distance(phash_val, ref_hash) <= threshold:
                print(f"Cross Perceptual Duplicate: {os.path.basename(filepath)} visually matches approved {os.path.basename(ref_path)}")
                if os.path.exists(filepath):
                    os.rename(filepath, os.path.join(discard_dir, os.path.basename(filepath)))
                    removed_count += 1
                break
                
    print(f"Cross-directory deduplication complete. Removed {removed_count} cross-duplicates.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean exact and near image duplicates.")
    parser.add_argument("--dir", type=str, default="data/scraped_approved", help="Primary directory to deduplicate")
    parser.add_argument("--ref-dir", type=str, default=None, help="Reference directory to check against for cross-deduplication")
    parser.add_argument("--threshold", type=int, default=2, help="Hamming distance threshold (0-64) for perceptual matching")
    args = parser.parse_args()
    
    # Run internal deduplication on the target directory first
    deduplicate_directory(args.dir, args.threshold)
    
    # Run cross-deduplication if reference dir is provided
    if args.ref_dir:
        deduplicate_cross_directories(args.dir, args.ref_dir, args.threshold)
