from typing import List, Union, Optional

from histoprocess import AnnotationType
from histoprocess._domain.model.polygon import GeometryType, Polygon, Polygons
from pydantic import BaseModel


class HandlerResponse(BaseModel):
    media: str
    polygons: List

    class Config:
        arbitrary_types_allowed = True


class PydenticPolygon(BaseModel):
    geometry_type: GeometryType
    annotation_type: AnnotationType
    color: Union[List, int]
    coordinates: List[List]
    name: Optional[str]

    @classmethod
    def form_polygon(cls, polygon: Polygon):
        return cls(
            geometry_type=polygon.geometry_type,
            annotation_type=polygon.annotation_type,
            color=polygon.color,
            coordinates=polygon.coordinates,
            name=polygon.name,
        )


class PydenticPolygons(BaseModel):
    value: List[PydenticPolygon]

    @classmethod
    def form_polygon(cls, polygons: Polygons):
        return cls(
            value=[PydenticPolygon.form_polygon(poly) for poly in polygons.value]
        )
