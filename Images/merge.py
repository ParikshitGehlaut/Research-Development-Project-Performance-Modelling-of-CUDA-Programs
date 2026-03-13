import os
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

ais = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
archs = ['RTX5000', 'A100', 'H100']

def get_image_path(arch, ai):
    base_dir = os.path.join(ROOT, 'Results', arch, 'synthetic_streaming')
    if arch == 'A100':
        return os.path.join(base_dir, f'synthetic_kernel_a{ai}_a100.png')
    elif arch == 'H100':
        return os.path.join(base_dir, f'synthetic_kernel_a{ai}_h100.png')
    elif arch == 'RTX5000':
        return os.path.join(base_dir, f'synthetic_kernel_a{ai}_rtx5000.png')
    return None

os.makedirs(SCRIPT_DIR, exist_ok=True)

for ai in ais:
    images = []
    missing_any = False
    for arch in archs:
        path = get_image_path(arch, ai)
        if path and os.path.exists(path):
            try:
                img = Image.open(path)
                images.append(img)
            except Exception as e:
                print(f"Error opening {path}: {e}")
                missing_any = True
        else:
            print(f"Missing image for AI={ai}, Arch={arch} at {path}")
            missing_any = True
            
    if len(images) == 3 and not missing_any:
        # Assuming all images have same height
        widths, heights = zip(*(i.size for i in images))
        
        total_width = sum(widths)
        max_height = max(heights)
        
        new_im = Image.new('RGB', (total_width, max_height), color='white')
        
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
            
        out_path = os.path.join(SCRIPT_DIR, f'ai{ai}_row_merged.png')
        new_im.save(out_path)
        print(f"Successfully merged {out_path}")
    else:
        print(f"Failed to merge for AI={ai} due to missing images.")
