import enum
from pathlib import Path
from typing import List, Optional
from src.data.data_builder import DataBuilder
from src.data.image_dataset import ImageDataset
from torch.utils.data.dataloader import DataLoader
from histoprocess._presentation.image.image_inference_collection import (
    ImageInferenceCollection,
)
from histoprocess._presentation.image.image_grid_feature import ImageGridFeature
from histoprocess.transforms import PatchTransformer


class GridType(enum.Enum):
    FULL = 0
    ANNOTATION = 1


class ImageDataBuilder(DataBuilder):

    def __init__(
        self,
        image_path: Path,
        grid_type: GridType,
        wsa_path: Optional[str] = None,
        transforms: Optional[List] = None,
        is_fill_without_mask=False,
        is_fill_without_tissue=False,
    ):
        self._grid = self._generate_grid(grid_type, image_path, wsa_path)
        collection = ImageInferenceCollection.init(
            grid=self._grid,
            image_path=str(image_path),
            is_fill_without_tissue=is_fill_without_tissue,
            is_fill_without_mask=is_fill_without_mask,
            transformer=(
                PatchTransformer.init(transforms=transforms)
                if transforms is not None
                else None
            ),
        )
        self.patches = [patch for patch in collection.get_patches_iterator()]
        self.images = [patch.tile for patch in self.patches]
        self.dataset = ImageDataset(image=self.images)

    def build(self):
        return DataLoader(
            dataset=self.dataset,
            num_workers=30,
            batch_size=10,
            pin_memory=False,
        )

    @property
    def grid(self):
        return self._grid

    def _generate_grid(
        self,
        grid_type: GridType,
        image_path: Path,
        wsa_path: Optional[str] = None,
    ):
        return (
            ImageGridFeature.init().get_full_grid(
                image_path=str(image_path),
                patch_size={"pixel": (512, 512)},
                overlap=256,
                percentage_tissue_in_tile=0.2,
            )
            if grid_type == GridType.FULL
            else ImageGridFeature.init().get_annotation_grid(
                image_path=str(image_path),
                wsa_path=wsa_path,
                patch_size={"pixel": (256, 256)},
                overlap=64,
                percentage_mask_in_tile=0.1,
            )
        )
