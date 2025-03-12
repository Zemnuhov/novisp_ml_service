from abc import ABC, abstractmethod
from histoprocess._domain.model.polygon import Polygons

class Predictor(ABC):

    @abstractmethod
    async def predict(self, *kwargs) -> Polygons:
        raise NotImplementedError()