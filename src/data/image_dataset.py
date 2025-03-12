from typing import List
from histoprocess._presentation.collections import Collection
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, image: List):
        super().__init__()
        self._images = image

    def __getitem__(self, index):
        return self._images[index]

    def __len__(self):
        return len(self._images)


class CollectionDataset(Dataset):

    def __init__(self, collection: Collection):
        super().__init__()
        self._collection = collection

    def __getitem__(self, index):
        return self._collection.get_patch_by_index(index).tile

    def __len__(self):
        return len(self._collection)
