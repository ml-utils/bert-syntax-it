from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import urlopen


def is_internet_on(url="https://huggingface.co"):
    try:
        _ = urlopen(url, timeout=2.50)
        return True
    except (URLError, HTTPError):
        return False
