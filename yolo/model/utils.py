import time
import functools

def time_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = (time.time()  - start_time) * 1000
        print(f"{func.__name__}, {duration:.4f}")
        return result
    return wrapper