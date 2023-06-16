#%%
from pathlib import Path
import subprocess
import sys

fpath = Path(__file__).parent.resolve()

print(fpath)
# %%

def execute(command_line):
    stdout = subprocess.run(command_line, check=True, capture_output=True) # Success!
    print(stdout)

data_path = fpath / 'data'
imgs_path = data_path / 'imgs'
predicted_imgs_path = data_path / 'imgs_predicted'
predicted_imgs_path.mkdir(exist_ok=True)

for p in imgs_path.iterdir():
    if p.is_file():
        print(p)
        command_line = [
            sys.executable,
            fpath / 'predict.py',
            '--input',
            p,
            '--output',
            predicted_imgs_path / (p.stem + "_out.png"),
            '--model',
            fpath / 'checkpoints' / 'checkpoint_epoch5.pth',
        ]
        execute(command_line)
        # break
# %%
