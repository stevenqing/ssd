import numpy as np


def moving_average_smooth_cost(data, window_size):
    smooth_data = []
    for i in range(len(data)):
        if i < window_size:
            smooth_data.append(sum(data[:i + 1]) / (i + 1))
        elif i >= len(data) - window_size:
            smooth_data.append(sum(data[i:]) / (len(data) - i))

        else:
            smooth_data.append(
                sum(data[i - window_size + 1:i + 1]) / window_size)
    return smooth_data


def moving_average_smooth(data, window_size):
    smooth_data = []
    for i in range(len(data)):
        if i < window_size:
            smooth_data.append(sum(data[:i + 1]) / (i + 1))
        elif i >= len(data) - window_size:
            smooth_data.append(sum(data[i:]) / (len(data) - i))

        else:
            smooth_data.append(
                sum(data[i - window_size + 1:i + 1]) / window_size)
    return smooth_data


def exponential_moving_average(data, alpha):
    data = np.array(data)
    multiplier = 1 - alpha
    ema = data[0]
    ema_values = np.zeros(len(data))
    ema_values[0] = ema
    for i in range(1, len(data)):
        ema = (alpha * data[i]) + (multiplier * ema)
        ema_values[i] = ema
    return ema_values.tolist()