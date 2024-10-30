# Импорт библиотек
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import pymc3 as pm
import arviz as az
import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy import stats
import theano
import theano.tensor as tt
from theano.tensor.signal.conv import conv2d

from covid.patients import get_delay_distribution


class GenerativeModel:
    version = "1.0.0"

    def __init__(self, country: str, observed: pd.DataFrame):

        self._trace = None
        self._inference_data = None
        self.observed = observed
        self.country = country

    def _get_generation_time_interval(self):
        """Создает дискретные P(Генерационный интервал)
        Source: https://www.ijidonline.com/article/S1201-9712(20)30119-3/pdf"""
        import scipy.stats as sps

        # Задаем параметры гамма-распределения
        shape = 2.6
        rate = 0.4

        # Создаем гамма-распределение
        # scale = 1 / rate, поскольку в scipy rate = 1/scale
        dist = sps.gamma(a=shape, scale=1/rate)

        # Дискретные генерационные интервалы максимально 20 дней (как будто скользящее окно размером 20)
        g_range = np.arange(0, 20)
        gt = pd.Series(dist.cdf(g_range), index=g_range)
        gt = gt.diff().fillna(0)
        gt /= gt.sum()
        gt = gt.values
        return gt

    def _get_convolution_ready_gt(self, len_observed):
        """Ускоряет theano.scan, предварительно вычисляя вектор интервала времени генерации. Спасибо Junpeng Lao за эту оптимизацию.
        Пожалуйста, ознакомьтесь с математикой моделирования вспышки здесь:
        https://staff.math.su.se/hoehle/blog/2020/04/15/effectiveR0.html"""
        gt = self._get_generation_time_interval()
        convolution_ready_gt = np.zeros((len_observed - 1, len_observed))
        for t in range(1, len_observed):
            begin = np.maximum(0, t - len(gt) + 1)
            slice_update = gt[1 : t - begin + 1][::-1]
            convolution_ready_gt[t - 1, begin : begin + len(slice_update)] = slice_update
        convolution_ready_gt = theano.shared(convolution_ready_gt)
        return convolution_ready_gt
    
    # В предыдущей выборке GaussianRandomWalk есть ошибка,
    # эта обезьяна-патчит исправление. Это должно быть исправлено скоро в PyMC3.
    def _random(self, sigma, mu, size, sample_shape):
        """Реализуйте гауссовское случайное блуждание как кумулятивную сумму нормалей."""
        if size[len(sample_shape)] == sample_shape:
            axis = len(sample_shape)
        else:
            axis = len(size) - 1
        rv = stats.norm(mu, sigma)
        data = rv.rvs(size).cumsum(axis=axis)
        data = np.array(data)
        if len(data.shape) > 1:
            for i in range(data.shape[0]):
                data[i] = data[i] - data[i][0]
        else:
            data = data - data[0]
        return data
    
    @staticmethod
    def conv(a, b, len_observed):
        """Perform a 1D convolution of a and b"""
        from theano.tensor.signal.conv import conv2d

        return conv2d(
            tt.reshape(a, (1, len_observed)),
            tt.reshape(b, (1, len(b))),
            border_mode="full",
        )[0, :len_observed]

    def build(self):
        """ Строит и возвращает генеративную модель. Также устанавливает self.model """
        
        # Получаем распределение задержек через гамма распределение
        inc = get_delay_distribution()
        inc = inc.values
        
        # Определеяем количество наблюдений
        len_observed = len(self.observed)
        
        # Подготавлимаем генеративные интервалы
        convolution_ready_gt = self._get_convolution_ready_gt(len_observed)
        
        # Координаты
        with pm.Model() as self.model:

            # Пусть log_r_t ходит случайным образом с фиксированным априором ~0,035. 
            # Думайте об этом числе как о том, как быстро r_t может реагировать.
            # В общем это блуждания
            pm.GaussianRandomWalk._random = self._random
            log_r_t = pm.GaussianRandomWalk(
                    "log_r_t",      # Название параметра для отслеживания
                    sigma=0.035,    # Стандартное отклонение для случайного блуждания, определяющее плавность изменений
                    shape=len_observed,  # Количество временных шагов (дней) для моделирования
                )
            
            # Преобразование log(R_t) в R_t, чтобы получить положительные значения
            r_t = pm.Deterministic("r_t", pm.math.exp(log_r_t))  # Определение R_t как экспоненты от log(R_t)

            # Для заданной популяции семян и кривой R_t мы вычисляем
            # предполагаемую кривую заражения, моделируя вспышку. Хотя это может
            # выглядеть пугающе, это просто способ воссоздать вспышку
            # математика моделирования внутри модели:
            # https://staff.math.su.se/hoehle/blog/2020/04/15/effectiveR0.html
            
            # Инициализация начальной популяции зараженных
            seed = pm.Exponential("seed", 0.01)  # Начальная популяция инфекций с экспоненциальным распределением
            y0 = tt.zeros(len_observed)  # Массив для хранения количества инфекций на каждом временном шаге
            y0 = tt.set_subtensor(y0[0], seed)  # Установка начального значения зараженных на первый день
            
            # Реализация рекурсивного алгоритма для моделирования инфекций с учетом генерационного интервала
            outputs, _ = theano.scan(
                fn=lambda t, gt, y, r_t: tt.set_subtensor(y[t], tt.sum(r_t * y * gt)),
                sequences=[tt.arange(1, len_observed), convolution_ready_gt],
                outputs_info=y0,  # начальное состояние y0 (число инфицированных на каждом шаге)
                non_sequences=r_t,  # параметр R_t, изменяющийся во времени
                n_steps=len_observed - 1,  # количество временных шагов, охватывающих все дни наблюдений
            )
            
            # Определяем infections как детерминированную переменную, представляющую количество инфекций
            infections = pm.Deterministic("infections", outputs[-1])  # Конечный результат обновления для infections
            
            # Ограничиваем предсказанное количество инфекций в диапазоне от 0 до 10 000 000
            infections = tt.clip(infections, 0, 10_000_000)
            
            # Определение веса для корректировки дневного эффекта
#             w = pm.Dirichlet("w", a=[1, 1, 1, 1, 1, 1, 1])
            
            # Свертывание инфекций в подтвержденные положительные отчеты на основе известного
            # распределения p_delay. Подробности о том, как мы вычисляем
            # это распределение, см. в patients.py.
            # Мы заменяем расчет p_delay на гамма распределение с заданными параметрами inc
            # Применяем свертку и корректируем `test_adjusted_positive` на основе недельного веса `w`
            test_adjusted_positive = pm.Deterministic(
                "test_adjusted_positive", self.conv(infections, inc, len_observed)
            )
            
#             # Итерация по дням с учетом весов Дирихле
#             weighted_output, _ = theano.scan(
#                 fn=lambda t, test_value, w: test_value * w[t % 7],
#                 sequences=[tt.arange(1, len_observed), test_adjusted_positive],
#                 non_sequences=w,
#                 n_steps=len_observed - 1,
#             )
            
#             # Сохраняем результирующие значения скорректированных инфекций
#             weighted_infections = pm.Deterministic("weighted_infections", weighted_output[-1])
            
            # Определение функции правдоподобия наблюдаемых данных с использованием отрицательного биномиального распределения
            pm.NegativeBinomial(
                "obs",  # Название вероятностного распределения для наблюдаемых данных
                test_adjusted_positive + 0.1,  # Добавляем небольшой оффсет, чтобы избежать значений 0 на начальных днях
                alpha=pm.Gamma("alpha", mu=6, sigma=1),  # Гиперпараметр alpha с гамма-распределением
                observed=self.observed.new_cases,  # Передаем реальные данные для обучения модели
            )

        return self.model
