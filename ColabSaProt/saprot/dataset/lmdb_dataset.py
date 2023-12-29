import abc
import torch
import lmdb
import pytorch_lightning as pl
import copy

from torch.utils.data import DataLoader
from tqdm import tqdm


_10TB = 10995116277760


class LMDBDataset(pl.LightningDataModule):
    """
    Abstract class from which other datasets inherit. We use LMDB database for all subclasses.
    """
    def __init__(self,
                 train_lmdb: str = None,
                 valid_lmdb: str = None,
                 test_lmdb: str = None,
                 dataloader_kwargs: dict = None):
        """
        Args:
            train_lmdb: path to train lmdb
            valid_lmdb: path to valid lmdb
            test_lmdb: path to test lmdb
            dataloader_kwargs: kwargs for dataloader
        """
        super().__init__()
        self.train_lmdb = train_lmdb
        self.valid_lmdb = valid_lmdb
        self.test_lmdb = test_lmdb
        self.dataloader_kwargs = dataloader_kwargs if dataloader_kwargs is not None else {}

        self.env = None
        self.operator = None
    
    def is_initialized(self):
        return self.env is not None
    
    def _init_lmdb(self, path):
        if self.env is not None:
            self._close_lmdb()
            
        # open lmdb
        self.env = lmdb.open(path, lock=False, map_size=_10TB)
        self.operator = self.env.begin()
    
    def _close_lmdb(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.operator = None
    
    def _cursor(self):
        return self.operator.cursor()

    def _get(self, key: str or int):
        value = self.operator.get(str(key).encode())
        
        if value is not None:
            value = value.decode()
        
        return value
     
    def _dataloader(self, stage):
        self.dataloader_kwargs["shuffle"] = True if stage == "train" else False
        lmdb_path = getattr(self, f"{stage}_lmdb")
        dataset = copy.copy(self)
        dataset._init_lmdb(lmdb_path)
        setattr(dataset, "stage", stage)
        
        return DataLoader(dataset, collate_fn=dataset.collate_fn, **self.dataloader_kwargs)
    
    def train_dataloader(self):
        return self._dataloader("train")

    def test_dataloader(self):
        return self._dataloader("test")
    
    def val_dataloader(self):
        return self._dataloader("valid")
        
    @abc.abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def collate_fn(self, batch):
        """
        Datasets should implement it as the function will be set when initializing Dataloader

        Returns:
            inputs: dict
            labels: dict
        """
        raise NotImplementedError