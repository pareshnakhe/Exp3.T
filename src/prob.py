import random


def draw(weights):
    """
    pick an index from the given list of floats proportional
    to the size of the entry (normalize to a probability
    distribution and draw according to the probabilities).
    :param weights:
    :return: index of action chosen
    """
    choice = random.uniform(0, sum(weights))
    choice_index = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choice_index

        choice_index += 1


def Bubeck_draw(weights, t=1):
    choice = random.uniform(0, sum(weights))
    choice_index = 0

    for weight in weights:
        choice -= weight
        if choice <= 0:
            return choice_index

        choice_index += 1


def distr(weights, gamma=0.0, rnd=1):
    """
    Normalize a list of floats to a probability distribution.
    Gamma is a factor, which tempers the distribution toward
    being uniform as it grows from zero to one.
    """
    the_sum = float(sum(weights))
    if rnd % 100000 == 0:
        for itr in range(len(weights)):
            weights[itr] /= the_sum

    return tuple((1.0 - gamma) * (w / the_sum) + (gamma / len(weights)) for w in weights)


def Bubeck_distr(weights):
    return tuple(w / float(sum(weights)) for w in weights)


def exp3S_distr(weights, gamma=1.0):
    return tuple((1.0 - gamma) * (w / float(sum(weights))) + (gamma / len(weights)) for w in weights)


def mean(a_list):
    tmp_sum = sum([x for x in a_list])
    return 0 if len(tmp_sum) == 0 else tmp_sum / len(tmp_sum)
