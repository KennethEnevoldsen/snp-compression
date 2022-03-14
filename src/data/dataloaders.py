import os
import random
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import dask.array as da

import torch
from torch.utils.data import Dataset, IterableDataset
import dask
from pandas_plink import read_plink1_bin, write_plink1_bin


import xarray as xr
from xarray import DataArray


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
        except StopIteration:
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


class PLINKIterableDataset(IterableDataset):
    """
    Creates a PLINK iterable dataset, which load the .bed files along with its metadata
    using XArray, which allow for loading the SNPs along with their metadata.

    Args:
        plink_path (Optional[str], optional): Path to the .bed or .zarr file. Defaults
            to None which corresponds to: os.path.join("/home", "kce", "dsmwpred",
            "data", "ukbb", "geno.bed"). If it is a .zarr file, it will load the
            "genotype" DataArray from the loaded Xarray dataset.
        buffer_size (int, optional): Defaults to 1024.
        chromosome (Optional[int], optional): Defaults to None,
            indicating all chromosomes.

    Returns:
        IterableDataset: An iterable dataset. Containin the genotype data.
    """

    def __init__(
        self,
        plink_path: str,
        buffer_size: int = 1024,
        shuffle: bool = True,
        chromosome: Optional[int] = None,
        seed: int = 42,
        to_tensor: bool = True,
    ) -> IterableDataset:
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.seed = seed
        self.convert_to_tensor = to_tensor

        self.from_disk(plink_path)
        self.set_chromosome(chromosome)

    def set_chromosome(self, chromosome: Optional[int]) -> None:
        """
        Set the chromosome to be loaded.
        """
        if chromosome:
            self.genotype = self._genotype.where(
                self._genotype.chrom == str(chromosome), drop=True
            )
        else:
            self.genotype = self._genotype

    def __create_iter(self) -> Iterator:
        n, _ = self._genotype.shape

        starts = range(0, n, self.buffer_size)
        ends = range(self.buffer_size, n, self.buffer_size)

        end = 0  # if buffer_size > array.shape[1]
        for start, end in zip(starts, ends):
            X = self._genotype[start:end]
            assert X.shape[0] == self.buffer_size
            for x in X:
                yield x
        if n > end:
            X = self._genotype[end:n]
            for x in X:
                yield x

    def create_data_array_iter(self) -> Iterator:
        """
        An iterable version of the plink dataset. If shuffle is True, the data is
        shuffled using a shufflebuffer.

        Yields:
            DataArray: A DataArray object containing the genotype data.
        """
        dataset_iter = self.__create_iter()
        if self.shuffle:
            dataset_iter = self.shuffle_buffer(dataset_iter)
        return dataset_iter

    def __iter__(self) -> Iterator:
        dataset_iter = self.create_data_array_iter()

        for x in dataset_iter:
            if self.convert_to_tensor:
                yield self.to_tensor(x)
            else:
                yield x

    def save_to_disk(self, path: str):
        """
        Save the dataset to disk.

        Args:
            path (str): Path to save the dataset. Save format is determined by the
                file extension. Options include ".bed" or ".zarr". Defaults to
                ".zarr".
        """
        ext = os.path.splitext(path)[-1]
        if ext == ".bed":
            write_plink1_bin(self._genotype, path)
        elif ext == ".zarr":
            ds = xr.Dataset(dict(genotype=self._genotype))
            ds.to_zarr(path)
        else:
            raise ValueError("Unknown file extension, should be .bed or .zarr")

    def from_disk(self, path: str) -> None:
        """
        Load the dataset from disk.

        Args:
            path (str): Path to the dataset. Read format is determined by the
                file extension. Options include ".bed" or ".zarr".

        """
        ext = os.path.splitext(path)[-1]
        if ext == ".bed":
            self._genotype = read_plink1_bin(path)
        elif ext == ".zarr":
            zarr_ds = xr.open_zarr(path)
            self._genotype = zarr_ds.genotype
        else:
            raise ValueError("Unknown file extension, should be .bed or .zarr")

    def to_tensor(self, x: DataArray):
        """Convert DataArray to tensor

        Args:
            x (DataArray): A DataArray object containing the genotype data.
        """
        return torch.from_numpy(x.compute().data)

    def shuffle_buffer(self, dataset_iter: Iterator) -> Iterator:
        """Creates a shuffle buffer for the dataset.

        Args:
            dataset_iter (Iterator): An iterator of the dataset
        """
        random.seed(self.seed)

        shufbuf = []
        try:
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except StopIteration:
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
    splits[n_val: n_val + n_test] = 2
    np.random.seed(1234)  # ensures consistent splits
    np.random.shuffle(splits)

    with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        train = arr[splits == 0]
        val = arr[splits == 1]
        test = arr[splits == 2]

    ds_train = DaskIterableDataset(array=train, buffer_size=buffer_size)
    ds_val = DaskIterableDataset(array=val, buffer_size=buffer_size)
    ds_test = DaskIterableDataset(array=test, buffer_size=buffer_size)
    return (ds_train, ds_val, ds_test)


if __name__ == "__main__":
    path = os.path.join("/home", "kce", "NLPPred", "data-science-exam", "mhcabi.bed")
    dataset = PLINKIterableDataset(path, chromosome=6, to_tensor=False)
    
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset)
    plink_iter = iter(dataset)
    sample = next(plink_iter)
    print(type(sample))


    # meta_data_iter = dataset.create_data_array_iter()
    # sample_meta = next(meta_data_iter)

    # meta_data_iter = dataset.create_data_array_iter()
    # sample_meta2 = next(meta_data_iter)
    # print("done")

    # load_dataset_with_bim_and_fam(chromosome=6)
    # import time

    # path = os.path.join("data", "interim", "chrom_1")
    # arr = da.from_zarr(path)  # shape (participants, snps)

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

#     dataset, val, test = load_dataset(22, 1024)
#     dl = DataLoader(dataset, batch_size=64, shuffle=False)
#     s = time.time()
#     iter_dl = iter(dl)
#     n = 0
#     for i, d in enumerate(iter(dl)):
#         n += d.shape[0]
#         assert isinstance(d, torch.Tensor)

# from xarray import DataArray
# import xarray as xr
# from pandas_plink import read_plink1_bin, write_plink1_bin
# import os

# # path = os.path.join("/home", "kce", "dsmwpred", "data", "ukbb", "geno.bed")
# # G = read_plink1_bin(path)
# # G


# D = DataArray(
#     [[3.0, 2.0, 2.0], [0.0, 0.0, 1.0]],
#     dims=["sample", "variant"],
#     coords=dict(
#         iid=("sample", [1, 2]),
#         chrom=("variant", ["1", "1", "2"]),
#     ),
# )

# C = DataArray(
#     [[3.0, 2.0, 2.0], [0.0, 0.0, 1.0]],
#     dims=["sample", "variant"],
#     coords=dict(
#         iid=("sample", [1, 2]),
#         chrom=("variant", ["1", "1", "2"]),
#     ),
# )

# # C = DataArray(
# #     [[3.0, 2.0, 2.0], [0.0, 0.0, 1.0]],
# #     dims=C.dims,
# #     coords=C.coords,
# # )


# # print(D)
# write_plink1_bin(D, "sample.bed", verbose=False)
# t = read_plink1_bin("sample.bed")
# t = xr.Dataset(dict(genotype=t))
# t.to_zarr("sample.zarr")

# ds_zarr = xr.open_zarr("sample.zarr")
# genotype = ds_zarr.genotype
# xarr = xr.open_dataset("saved_on_disk.nc").to_array()

# xr.concat([D, C], dim="sample")
