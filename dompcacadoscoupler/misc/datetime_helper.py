import datetime


def get_time_string_now(time_format: str = '%Y%m%d_%H%M%S') -> str:
    time_string_now = datetime.datetime.now().strftime(time_format)
    return time_string_now