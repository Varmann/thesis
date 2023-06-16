#%%
from pathlib import Path

fpath = Path(__file__).parent.resolve()

data_path = fpath / 'data'
imgs_path = data_path / 'imgs'
predicted_imgs_path = data_path / 'imgs_predicted'
predicted_imgs_path.mkdir(exist_ok=True)

image_file_paths = [str(p) for p in imgs_path.iterdir() if p.is_file()]
predicted_images_file_paths = [str(predicted_imgs_path / (p.stem + "_out.png")) for p in imgs_path.iterdir() if p.is_file()]

print(image_file_paths)
print(predicted_images_file_paths)
# %%
