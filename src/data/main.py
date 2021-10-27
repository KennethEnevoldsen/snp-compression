import os

from dataloader import write_plink_to_pt

# Old data
write_plink_to_pt(
    plink_path=os.path.join("data", "raw", "mhcuvps"),
    save_path=os.path.join("data", "processed", "tensors"),
)

