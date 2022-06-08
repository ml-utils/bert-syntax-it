from pathlib import Path


def get_int_tests_root() -> Path:
    return Path(__file__).parent


def get_test_data_dir() -> Path:
    return get_int_tests_root() / "test_data"
