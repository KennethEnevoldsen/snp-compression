import os
import random
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import dask.array as da

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


class DaskMapDataset(Dataset):
    """
    MAP implementation of torch dataset, for dask array.
    Very inefficient due to individual computes pr. idx.
    Takes about 800s pr. 1k samples on chrom 1.
    """

    def __init__(self, array: da.Array) -> Dataset:
        self.array = array

    def __len__(self):
        return self.array.shape[0]

    def __getitem__(self, idx: int):
        return self.array[idx].compute()


class DaskIterableDataset(IterableDataset):
    """
    An iterable version of the dask dataset for torch. This is much faster about
    at less than 1s pr. 1k samples (800 times faster than map).
    """

    def __init__(self, array: da.Array, buffer_size: int = 1024) -> IterableDataset:
        self.array = array
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator:
        n, _ = self.array.shape

        starts = range(0, n, self.buffer_size)
        ends = range(self.buffer_size, n, self.buffer_size)

        end = 0  # if buffer_size > array.shape[1]
        for start, end in zip(starts, ends):
            X = torch.from_numpy(self.array[start:end].compute())
            assert X.shape[0] == self.buffer_size
            for x in X:
                yield x
        if n > end:
            X = torch.from_numpy(self.array[end:n].compute())
            for x in X:
                yield x


class ShuffleDataset(IterableDataset):
    """
    An dataset wrapper for adding a shufflebuffer to an iterable dataset.
    """

    def __init__(self, dataset, buffer_size: int = 1024):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass


def load_dataset(
    chromosome: int,
    buffer_size: int = 1024,
    p_val: Union[float, int] = 10_000,
    p_test: Union[float, int] = 10_000,
    limit_train: Optional[int] = None,
) -> Tuple[ShuffleDataset, ShuffleDataset, ShuffleDataset]:
    """Load the specific chromosome required

    Args:
        p_test (Union[float, int], optional): The proportion (float) or number (int) of
            the dataset which should be the test set.
        limit_train (Optional[int], optional): The limit of the number of training
            samples. Defaults to None indicating no limit.
    """
    f_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(f_path, "..", "..", "data", "interim", f"chrom_{chromosome}")
    arr = da.from_zarr(path)
    if limit_train:
        arr = arr[:limit_train, :]

    n_val = int(arr.shape[0] // (1 / p_val)) if isinstance(p_val, float) else p_val
    n_test = int(arr.shape[0] // (1 / p_test)) if isinstance(p_test, float) else p_val

    splits = np.zeros((arr.shape[0],))
    splits[:n_val] = 1
    splits[n_val : n_val + n_test] = 2
    np.random.seed(1234)  # ensures consistent splits
    np.random.shuffle(splits)

    train = arr[splits == 0].rechunk()
    val = arr[splits == 1].rechunk()
    test = arr[splits == 2].rechunk()

    ds_train = DaskIterableDataset(array=train, buffer_size=buffer_size)
    ds_val = DaskIterableDataset(array=val, buffer_size=buffer_size)
    ds_test = DaskIterableDataset(array=test, buffer_size=buffer_size)
    return (ds_train, ds_val, ds_test)


if __name__ == "__main__":
    import time

    path = os.path.join("data", "interim", "chrom_1")
    arr = da.from_zarr(path)  # shape (participants, snps)

    # testing the speeds of the different datasets:

    # Iterable dataset
    # dataset = DaskIterableDataset(array=arr)
    # dl = DataLoader(dataset, batch_size=64, shuffle=False)
    # s = time.time()
    # iter_dl = iter(dl)
    # n = 0
    # for i, d in enumerate(iter(dl)):
    #     n += d.shape[0]
    #     assert isinstance(d, torch.Tensor)
    #     assert n > 0
    #     if i % 100 == 0 and i != 0:
    #         print(f"\t{n}: {(time.time() - s)/n*1000} sec/ 1k samples")
    #     if i > 200:
    #         break
    # print(f"iterable took {time.time() - s}")

    # .. with buffer
    # dataset = DaskIterableDataset(array=arr)
    # dataset = ShuffleDataset(dataset, 1024)
    # dl = DataLoader(dataset, batch_size=64, shuffle=False)
    # s = time.time()
    # iter_dl = iter(dl)
    # n = 0
    # for i, d in enumerate(iter(dl)):
    #     n += d.shape[0]
    #     assert isinstance(d, torch.Tensor)
    #     assert n > 0
    #     if i % 100 == 0 and i != 0:
    #         print(f"\t{n}: {(time.time() - s)/n*1000} sec/ 1k samples")
    #     if i > 200:
    #         break
    # print(f"iterable w. shuffle took {time.time() - s}")

    # map style dataset (allows for a 'true' shuffle but is MUCH slower)
    # dataset = DaskMapDataset(array=arr)
    # dl = DataLoader(dataset, batch_size=64, shuffle=True)
    # s = time.time()
    # iter_dl = iter(dl)
    # n = 0
    # for i, d in enumerate(iter(dl)):
    #     n += d.shape[0]
    #     assert isinstance(d, torch.Tensor)
    #     assert n > 0
    #     print(f"\t{n}: {(time.time() - s)/n*1000} sec/ 1k samples")
    #     if i > 200:
    #         break
    # print(f"map took {time.time() - s}")

    dataset, val, test = load_dataset(22, 1024)
    dl = DataLoader(dataset, batch_size=64, shuffle=False)
    s = time.time()
    iter_dl = iter(dl)
    n = 0
    for i, d in enumerate(iter(dl)):
        n += d.shape[0]
        assert isinstance(d, torch.Tensor)
