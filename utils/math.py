from functools import reduce

def prod(data):
    return int(reduce(lambda a,b: a*b, data, 1))

def argsort(data):
    return sorted(range(len(data)), key=data.__getitem__)
