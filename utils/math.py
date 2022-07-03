from functools import reduce

def prod(data):
    return int(reduce(lambda a,b: a*b, data, 1))
