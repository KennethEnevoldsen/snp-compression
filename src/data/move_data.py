"""
"""

import os
import shutil
from pathlib import Path

original = os.path.join(Path.home(), "NLPPred")
files = [os.path.join(original, f) for f in os.listdir(original) if f.lower().endswith(".bed")]
target = os.path.join(Path.home(), "github", "snp-compression", "data", "raw")

for f in files:
    shutil.move(f, target)

print("[INFO] Files in raw:")
print(os.listdir(target))

files = [os.path.join(target, f) for f in os.listdir(target) if f.lower().endswith(".bed")]


with open(files[0], "rb") as f:
    text = f.read()

text.decode(errors="ignore")[:100]

import ast
import csv


def _parse_bytes(field):
    """ Convert string represented in Python byte-string literal b'' syntax into
        a decoded character string - otherwise return it unchanged.
    """
    result = field
    try:
        result = ast.literal_eval(field)
    finally:
        return result.decode() if isinstance(result, bytes) else field


def my_csv_reader(filename, /, **kwargs):
    with open(filename, 'rt', newline='') as file:
        for row in csv.reader(file, **kwargs):
            yield [_parse_bytes(field) for field in row]


reader = my_csv_reader(files[0], delimiter='\t')
for row in reader:
    print(row)

