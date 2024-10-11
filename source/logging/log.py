from enum import Enum, auto
from datetime import datetime

class LogChannels(Enum):
    MASKS = auto()
    DIMENSIONS = auto()
    PADDING = auto()
    INIT = auto()
    DATA = auto()
    ENCODER = auto()
    DECODER = auto()
    TRAINING = auto()
    DEBUG = auto()
    PARAMS = auto()
    LOSSES = auto()

class Logger():

    channels: set[LogChannels]

    def __init__(self) -> None:
        self.channels = set()

    def add_log_channel(self, channel: LogChannels):
        """Add a channel that will be displayed by the logger"""
        self.channels.add(channel)
    
    def log(self, channel: LogChannels, message: str):
        """Log a message. The message will be displayed if and only if the channel is activated"""
        if(channel in self.channels):
            print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {channel.name.upper()}: {message}")

#Export as pseudo-singleton, any class can directly access the same instance of the logger
logger = Logger()