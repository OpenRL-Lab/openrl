from abc import ABC, abstractmethod


class BaseDriver(ABC):
    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError
