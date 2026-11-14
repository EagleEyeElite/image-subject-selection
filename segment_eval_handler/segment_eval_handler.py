import abc


class SegmentEvalHandler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, prompt) -> dict:
        pass
