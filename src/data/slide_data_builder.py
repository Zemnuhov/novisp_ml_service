import enum
from pathlib import Path
from typing import List, Optional

from histoprocess import InjectionSetting
from histoprocess._presentation.collections import RuntimeCollection
from histoprocess._presentation.grid_feature import GridFeature
from src.data.data_builder import DataBuilder
from src.data.image_dataset import CollectionDataset
from torch.utils.data.dataloader import DataLoader
from histoprocess.transforms import PatchTransformer


class GridType(enum.Enum):
    FULL = 0
    ANNOTATION = 1
    BBOX = 2


class SlideDataBuilder(DataBuilder):

    def __init__(
        self,
        wsi_path: Path,
        grid_type: GridType,
        wsa_path: Optional[Path] = None,
        transforms: Optional[List] = None,
        is_fill_without_mask=False,
        is_fill_without_tissue=False,
    ):
        InjectionSetting().set_backend("cucim")
        self._grid = self._generate_grid(grid_type, wsi_path, wsa_path)
        collection = RuntimeCollection.init(
            grid=self._grid,
            wsi_path=str(wsi_path),
            wsa_path=str(wsa_path) if isinstance(wsa_path, Path) else wsa_path,
            is_fill_without_tissue=is_fill_without_tissue,
            is_fill_without_mask=is_fill_without_mask,
            transformer=(
                PatchTransformer.init(transforms=transforms)
                if transforms is not None
                else None
            ),
        )
        self.dataset = CollectionDataset(collection=collection)

    def build(self):
        return DataLoader(
            dataset=self.dataset,
            num_workers=2,
            batch_size=2,
            pin_memory=True,
            prefetch_factor=5,
        )

    @property
    def grid(self):
        return self._grid

    def _generate_grid(
        self,
        grid_type: GridType,
        wsi_path: Path,
        wsa_path: Optional[Path] = None,
    ):
        return (
            GridFeature.init().get_full_grid(
                wsi_path=str(wsi_path),
                patch_size={"pixel": (2048, 2048)},
                level=0,
                overlap=1024,
                percentage_tissue_in_tile=0.2,
            )
            if grid_type == GridType.FULL
            else (
                GridFeature.init().get_bbox_grid(
                    wsi_path=str(wsi_path),
                    wsa_path=str(wsa_path),
                    level=0,
                    padding_percentage=0.7,
                )
                if grid_type == GridType.BBOX
                else GridFeature.init().get_annotation_grid(
                    wsi_path=str(wsi_path),
                    wsa_path=str(wsa_path),
                    level=0,
                    patch_size={"pixel": (256, 256)},
                    overlap=64,
                )
            )
        )
