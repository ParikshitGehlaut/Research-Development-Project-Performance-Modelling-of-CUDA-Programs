import os
from PIL import Image, ImageChops

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

archs = ['RTX5000', 'A100', 'H100']

def get_mem_image_path(arch):
    return os.path.join(ROOT, 'Results', arch, 'MemLatency', f'mem_latency_heatmap_{arch.lower()}.png')

def get_arith_image_path(arch):
    return os.path.join(ROOT, 'Results', arch, 'ArithLatency', f'alu_latency_heatmap_{arch.lower()}.png')

def trim_vertical(im, padding=40):
    bg = Image.new(im.mode, im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        left, upper, right, lower = bbox
        return im.crop((0, max(0, upper - padding), im.size[0], min(im.size[1], lower + padding)))
    return im

def merge_images(paths, out_path, is_mem=True):
    images = []
    missing = False
    for p in paths:
        if os.path.exists(p):
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                print(f"Error opening {p}: {e}")
                missing = True
        else:
            print(f"Missing image at {p}")
            missing = True
            
    if missing or len(images) != 3:
        print(f"Failed to merge {out_path} due to missing images.")
        return
        
    # We use exact mathematical pixel boundaries because the canvas is hard-locked.
    # RTX5000 (0): Left margin is 875px. We slice off 250px of empty trailing white space.
    #              Right margin is 0.5" = 175px. We slice off 125px from right.
    # A100 (1): Left margin is 2.5" = 875px. We slice off 600px from left to erase label.
    #           Right margin is 175px. We slice off 125px from right.
    # H100 (2): Left margin is 875px. Slice off 600px from left.
    #           Right margin is 175px. We slice off 100px from right leaving 75px padding for "96" tick.
    
    # RTX 5000
    w0, h0 = images[0].size
    images[0] = images[0].crop((250, 0, w0 - 125, h0))
    
    # A100
    w1, h1 = images[1].size
    images[1] = images[1].crop((600, 0, w1 - 125, h1))
    
    # H100
    w2, h2 = images[2].size
    images[2] = images[2].crop((600, 0, w2 - 100, h2)) 
    
    total_width = sum(im.size[0] for im in images)
    max_height = max(im.size[1] for im in images)

    new_im = Image.new('RGB', (total_width, max_height), color='white')
    
    x_offset = 0
    # No extra padding_between needed because we left 50px of the native 175px right margin on the images
    for i, im in enumerate(images):
        y_offset = (max_height - im.size[1]) // 2 
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
        
    # Trim the top and bottom whitespace of the combined image 
    # to maximize vertical scaling when sized to \linewidth in LaTeX
    final_im = trim_vertical(new_im, padding=40)
    
    final_im.save(out_path)
    print(f"Successfully merged: {out_path}")

if __name__ == "__main__":
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    
    # Merge Mem Latency
    mem_paths = [get_mem_image_path(a) for a in archs]
    out_mem = os.path.join(SCRIPT_DIR, 'mem_latency_heatmap_merged.png')
    print("Merging Memory Latency...")
    merge_images(mem_paths, out_mem, is_mem=True)
    
    # Merge Arith Latency
    arith_paths = [get_arith_image_path(a) for a in archs]
    out_arith = os.path.join(SCRIPT_DIR, 'alu_latency_heatmap_merged.png')
    print("Merging Arithmetic Latency...")
    merge_images(arith_paths, out_arith, is_mem=False)
