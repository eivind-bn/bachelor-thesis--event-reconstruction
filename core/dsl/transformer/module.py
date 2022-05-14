from abc import ABC

from core.dsl.sink.module import Sink
from core.dsl.source.module import Source


class Transformer(Source, Sink, ABC):
    pass
