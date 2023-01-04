# abstract

from abc import ABC, abstractmethod

# import numpy as np


class Basic(ABC):

    @abstractmethod
    def hello(self):
        pass

# multiple inheritance


class Auto:
    def ride(self):
        print("Riding on a ground")


class Boat:
    def swim(self):
        print("Sailing in the ocean")


class Amphibian(Auto, Boat):
    pass


a = Amphibian()
a.ride()
a.swim()
