from abc import ABC, abstractmethod


class BaseBlurrer(ABC):
    @abstractmethod
    def blur(self, image, face, blur_strength=1.0):
        pass

    def get_name(self):
        return self.__class__.__name__

    def get_params(self):
        return {}
