
class Broadcaster:

    def __init__(self):
        self.subscriptions = {}

        self.block_dispatch = False
        self.nodes = set()

        self.properties = {}

    def notify(self, subject, **kwargs):
        if self.block_dispatch:
            return

        callbacks = self.subscriptions.get(subject)
        if callbacks is None:
            return
        else:
            for callback in callbacks:
                callback(**kwargs)

        self.block_dispatch = True
        for broadcaster in self.nodes:
            broadcaster.notify(subject, **kwargs)

        self.block_dispatch = False

    def subscribe(self, subject, callback):
        callbacks = self.subscriptions.get(subject)

        if callbacks is None:
            callbacks = [callback]
            self.subscriptions[subject] = callbacks
        else:
            callbacks.append(callback)

    def unsubscribe(self, subject, callback):
        callbacks = self.subscriptions.get(subject)

        if callbacks is None:
            return
        else:
            callbacks.remove(callback)

    def join(self, broadcaster):
        self.nodes.add(broadcaster)
        broadcaster.nodes.add(self)
