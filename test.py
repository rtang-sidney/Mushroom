import numpy as np


def analyser(number):
    analyser_x = np.random.rand(number)
    analyser_y = np.random.rand(number)
    return analyser_x, analyser_y


num = 100
analysers = analyser(number=num)
print(analysers[0][0])
