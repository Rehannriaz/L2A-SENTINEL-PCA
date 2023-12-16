import os
import rasterio
import numpy as np


class DataLoader:
    def __init__(self) -> None:
        return
    
    def loadFromList(paths: list(str)) -> np.ndarray:
        return np.asarray([rasterio.open(path).read(1) for path in paths])
    
    def loadFromPath(path: str) -> np.ndarray:
        return rasterio.open(path).read(1)
    
    def loadFromFolder(dirPath: str) -> np.ndarray:
        paths = os.listdir(path=dirPath)
        return np.asarray([rasterio.open(dirPath+'/'+path).read(1) for path in paths])