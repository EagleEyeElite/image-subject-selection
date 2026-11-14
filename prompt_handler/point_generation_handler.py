import abc


class PointGenerationHandler(metaclass=abc.ABCMeta):
    """
    method that generates points from a heatmap
    Returns:
        dict of points and point_labels
        e.g.
        np.array([[500, 375], [1125, 625]])
        np.array([1, 1])
    """

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, prompt):
        pass
