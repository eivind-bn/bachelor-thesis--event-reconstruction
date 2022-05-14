class EventBroadcaster:

    def __init__(self):
        self.subscriptions = {}
        self.nodes = set()
        self.dispatch_lock = set()

        self.key_value_store = {}

    def notify(self, subject, **kwargs):
        if subject in self.dispatch_lock:
            return

        callbacks = self.subscriptions.get(subject)
        if callbacks is not None:
            for callback in callbacks:
                callback(**kwargs)

        self.dispatch_lock.add(subject)
        for broadcaster in self.nodes:
            broadcaster.notify(subject, **kwargs)

        self.dispatch_lock.remove(subject)

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
