from abc import ABC

from core.sink.module import Sink
from core.source.module import Source


class Transformer(Source, Sink, ABC):
    pass
