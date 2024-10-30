import os
from scipy import stats as sps
import numpy as np
import pandas as pd



def get_delay_distribution():
    """ Возвращает эмпирическое распределение задержки между появлением симптома и подтвержденным положительным случаем. """

    # Задаем параметры гамма-распределения
    shape = 1.352
    rate = 0.265
    
    # Создаем гамма-распределение
    # scale = 1 / rate, поскольку в scipy rate = 1/scale
    dist = sps.gamma(a=shape, scale=1/rate)
    
    # Дискретные генерационные интервалы максимально 20 дней (как будто скользящее окно размером 20)
    inc_range = np.arange(0, 30)
    inc = pd.Series(dist.cdf(inc_range), index=inc_range)

    return inc
