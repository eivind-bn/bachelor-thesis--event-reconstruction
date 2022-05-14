from core.event.event_broadcaster import EventBroadcaster


class EventDispatcher:

    def __init__(self):
        self.message_dispatcher = EventBroadcaster()
