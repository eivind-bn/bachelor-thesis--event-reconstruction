from core.sink.module import Sink


class Printer(Sink):

    def __init__(self):
        self.name = 'foo'

    def process_data(self, height, width, data):
        print('init')
        print(self.name)

        def post_init(height, width, data):
            print('post_init')
            print(self.name)
            self.name = 'bar'


        self.process_data = post_init