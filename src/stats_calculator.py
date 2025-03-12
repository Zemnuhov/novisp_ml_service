from dataclasses import dataclass
from histoprocess._domain.model.polygon import Polygons
from histoprocess import AnnotationType


@dataclass
class Statistic:
    vessels_count: int
    clean_vessels: int
    invasion_vessels: int

    def to_string(self):
        return f"Всего сосудов: {self.vessels_count}\nЧистых сосудов: {self.clean_vessels}\nИнвазивных: {self.invasion_vessels}"


class StatisticCalculator:

    def get_stats_from_polygons(self, polygons: Polygons) -> Statistic:
        clean_counter = 0
        invasion_counter = 0
        for polygon in polygons.value:
            if polygon.annotation_type == AnnotationType.CLEAN_VESSEL:
                clean_counter += 1
            else:
                invasion_counter += 1
        return Statistic(
            vessels_count=len(polygons.value),
            clean_vessels=clean_counter,
            invasion_vessels=invasion_counter,
        )
