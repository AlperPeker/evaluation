import sys

VERBOSE = False

_USE_COLOR = sys.stdout.isatty()

_CYAN = "\033[36m"
_GREEN = "\033[32m"
_RESET = "\033[0m"


def set_verbose(v: bool) -> None:
    global VERBOSE
    VERBOSE = v


def _color(code: str, msg: str) -> str:
    if _USE_COLOR:
        return f"{code}{msg}{_RESET}"
    return msg


def info(msg: str) -> None:
    print(_color(_CYAN, msg))


def result(msg: str) -> None:
    print(_color(_GREEN, msg))
