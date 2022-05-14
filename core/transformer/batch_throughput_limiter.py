import time

from core.transformer.module import Transformer


class BSync(Transformer):

    def __init__(self, batch_per_second):
        super().__init__()

        self.batch_per_nanosecond = batch_per_second*1e-9
        self.batch_period = 1/self.batch_per_nanosecond
        self.t_mark = time.time_ns()

    def sleep_if_ahead_schedule(self):
        delta_time_ns = time.time_ns() - self.t_mark
        self.t_mark += delta_time_ns
        sleep_time = self.batch_period - delta_time_ns
        if sleep_time > 0:
            time.sleep(sleep_time*1e-9)

    def process_data(self, data, **kwargs):
        self.sleep_if_ahead_schedule()
        self.callback(data)