import os

from .data_handlers import write_plink_to_pt, write_plink_to_pt_batched

# Old data
write_plink_to_pt(
    plink_path=os.path.join("data", "raw", "mhcuvps"),
    save_path=os.path.join("data", "processed", "tensors"),
)

write_plink_to_pt_batched()
    plink_path = os.path.join("..", "..", "dsmwpred", "data", "ukbb", "geno")
    save_path = os.path.join("data", "processed", "tensors", "dsmwpred")
)