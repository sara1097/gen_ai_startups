from abc import ABC, abstractmethod

class BaseLLM(ABC):

    @abstractmethod
    def generate(self, message: list):
        pass

    @abstractmethod
    def stream(self, message : list):
        pass