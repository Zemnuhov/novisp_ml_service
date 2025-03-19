from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class DataBuilder(ABC):

    @abstractmethod
    def build(self) -> DataLoader:
        raise NotImplementedError()
