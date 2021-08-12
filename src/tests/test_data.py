from ..data.dataloader import read_plink_as_tensor
import os

def test_read_plink_as_tensor():
    file = os.path.join("data", "raw", "mhcuvps")
    x, y = read_plink_as_tensor(file)
    assert x.shape[1] == y.shape[0]


