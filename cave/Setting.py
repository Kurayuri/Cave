from .import CONSTANT
LogLevel = CONSTANT.INFO

def set_LogLevel(level):
    global LogLevel
    prev_level = LogLevel
    LogLevel = level
    return prev_level
