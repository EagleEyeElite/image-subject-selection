import abc
import pandas as pd
from PIL import Image
from utils.prompt import Prompt

# Abstract base class for handling datasets


class DatasetHandler(metaclass=abc.ABCMeta):
    def __init__(self, device):
        self.device = device

    # Abstract method to create a dataframe for the dataset
    @abc.abstractmethod
    def create_dataframe(self) -> pd.DataFrame:
        pass

    # Abstract method to get an image tensor from a dataset row
    @abc.abstractmethod
    def get_image_tensor(self, row, size=224):
        pass

    # Abstract method to get a PIL image from a dataset row
    @abc.abstractmethod
    def get_PIL_image(self, row, size=224) -> Image.Image:
        pass

    # Abstract method to get a mask from a dataset row
    @abc.abstractmethod
    def get_mask(self, row, size=224):
        pass

    # Abstract method to get the true label from a dataset row
    @abc.abstractmethod
    def get_true_label(self, row):
        pass

    # Method to get a sample of rows from the dataframe and convert them to Prompt objects
    def get_samples(self, sample_size):
        sample = self.dataframe.sample(sample_size)
        res = []
        for i in sample.itertuples():
            res.append(self.get_prompt(i))
        return res

    # Method to get all rows from the dataframe, or a sample if specified, as iterable tuples
    def get_all_rows(self, sample_size=None):
        if sample_size is not None:
            return self.dataframe.sample(sample_size).itertuples()
        return self.dataframe.copy().itertuples()

    # Method to create a Prompt object from a dataset row
    def get_prompt(self, row):
        p = Prompt()
        p.original_image_id = row.Index
        p.original_image_class = self.get_true_label(row)
        p.original_image_tensor = self.get_image_tensor(row)
        p.original_image = self.get_PIL_image(row)
        p.mask = self.get_mask(row)
        return p
