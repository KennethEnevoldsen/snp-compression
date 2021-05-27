"""
This script move the data to the right folder 
"""


import os
import shutil
from pathlib import Path

os.chdir("/home/kce/github/snp-compression")
origin = os.path.join(Path.home(), "NLPPred")
files = [f for f in os.listdir(origin) if f.split(".")[-1] in {"bed", "fam", "bim"}]

for f in files:
    src = os.path.join(origin, f)
    tgt = os.path.join("data", "raw", f)
    shutil.copy(src, tgt)
