from typing import Any, Dict, Optional, Tuple
import torch
import albumentations as A
from lightning import LightningDataModule
from src.data.components.lsp import LSP_Data
from src.data.components.transformed_dataset import transformed_dataset
from torch.utils.data import DataLoader, Dataset, random_split

class LSPDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/lsp",
        train_val_test_split: Tuple[int, int, int] = (7999, 2000, 1),
        batch_size: int = 16,
        num_workers: int = 4,
        sigma: int = 3,
        stride: int = 4,
        pin_memory: bool = False,
        train_transform: Optional[A.Compose] = None,
        val_test_transform: Optional[A.Compose] = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_transform = train_transform
        self.val_test_transform = val_test_transform

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            lsp = LSP_Data(self.hparams.data_dir)            
            
            train, val, test = random_split(
                dataset=lsp,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            self.data_train = transformed_dataset(train, self.hparams.stride, self.hparams.sigma, transform=self.train_transform)
            self.data_val = transformed_dataset(val, self.hparams.stride, self.hparams.sigma, transform=self.val_test_transform)
            self.data_test = transformed_dataset(test, self.hparams.stride, self.hparams.sigma, transform=self.val_test_transform)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    _ = LSPDataModule()
