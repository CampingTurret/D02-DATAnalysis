from tqdm.auto import tqdm,trange
import numpy as np
import pandas as pd

def GapDetect(DirtyData):
    """
    Removes Gaps in data due to errors in data aquisition
    """

    CleanData = DirtyData

    return CleanData