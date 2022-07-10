import time

def timer(func):
    def wrapper(*args, **kwargs):
        ts = time.time()
        ret = func(*args, **kwargs)
        cost = time.time() - ts
        return ret, cost
    return wrapper

