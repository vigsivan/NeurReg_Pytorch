import os
from typing import List
from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk

rootdir = Path("/Volumes/Untitled/spine-generic-multisubject/anat/t1/")
max_tup: List[int] = [0, 0, 0]
for i in tqdm(os.listdir(rootdir)):
    if not i.endswith("nii.gz"):
        continue
    image = sitk.ReadImage(str(rootdir / i))
    size = image.GetSize()
    for i in range(3):
        max_tup[i] = int(max(max_tup[i], size[i]))

print(max_tup)
