import abc


class ModelHandler(metaclass=abc.ABCMeta):
    def __init__(self, device, layer):
        # The device (e.g., CPU or GPU) where the model will be loaded
        self.device = device
        # The specific layer of the model for which we are interested in extracting features
        self.layer = layer
        self.model = self.load_model()  # Load the model by calling the abstract method
        # Load the input size by calling the abstract method
        self.input_size = self.load_input_size()
        self.labels = self.load_labels()  # Load the labels by calling the abstract method

    @abc.abstractmethod
    def load_model(self) -> any:
        # Abstract method to load the model
        pass

    @abc.abstractmethod
    def load_labels(self) -> list[str]:
        # Abstract method to load the labels
        pass

    @abc.abstractmethod
    def load_input_size(self) -> int:
        # Abstract method to load the input size
        pass
