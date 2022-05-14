from core.transformer.module import Transformer


class Mp4ToSingularEvents(Transformer):

    def __init__(self):
        super().__init__()

        self.old_levels = None

    def process_data(self, image, **kwargs):
        pass