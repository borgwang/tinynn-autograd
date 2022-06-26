from functools import reduce


def prod(data):
    return reduce(lambda a, b: a * b, data)


