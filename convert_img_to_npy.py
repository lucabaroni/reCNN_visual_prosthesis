#%%
from pathlib import Path
import numpy as np
from PIL import Image

src_dir = Path("v1_data/images")
dst_dir = Path("v1_data/images_npy")
dst_dir.mkdir(parents=True, exist_ok=True)

for png_path in sorted(src_dir.glob("*.png")):
    print(png_path)
    img = Image.open(png_path).convert("L")  # convert to grayscale (1 channel)
    arr = np.asarray(img, dtype=np.float32)    # or dtype=np.uint8 if preferred
    npy_path = dst_dir / (png_path.stem + ".npy")
    np.save(npy_path, arr, allow_pickle=False)
    print(f"Saved {npy_path}")
# %%
