"""Unit helpers. The engine works in bytes; results are reported in decimal GB and binary GiB."""

GB = 1e9
GIB = 1024**3


def gb(num_bytes: float) -> float:
    return num_bytes / GB


def gib(num_bytes: float) -> float:
    return num_bytes / GIB


def r(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def fmt_tokens(tokens: int) -> str:
    if tokens >= 1024 * 1024 and tokens % (1024 * 1024) == 0:
        return f"{tokens // (1024 * 1024)}M"
    if tokens >= 1024 and tokens % 1024 == 0:
        return f"{tokens // 1024}K"
    return f"{tokens:,}"
