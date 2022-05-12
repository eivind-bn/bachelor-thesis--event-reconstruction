from metavision_core.event_io import EventsIterator
from metavision_sdk_ui import Window, BaseWindow, UIAction, UIKeyEvent, EventLoop

from core.algorithm import Algorithm


class InteractiveWindow:

    def __init__(self,
                 title: str,
                 source: str,
                 algorithm_type: type(Algorithm),
                 delta_t=10000,
                 on_mouse_release=lambda alg, pos_log: {},
                 color_palette=BaseWindow.RenderMode.BGR,
                 **kwargs):
        event_stream = EventsIterator(input_path=source, delta_t=delta_t)
        height, width = event_stream.get_size()

        algorithm = algorithm_type(width, height, **kwargs)

        self.is_paused = False

        with Window(title=title, width=width, height=height, mode=color_palette) as window:

            def keyboard_cb(key, scancode, action, mods):
                if action != UIAction.RELEASE:
                    return
                if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                    window.set_close_flag()
                if key == UIKeyEvent.KEY_SPACE:
                    self.is_paused = not self.is_paused

            cursor_log = [[0, 0]] * 10

            def mouse_cb(key, action, a):
                if action == UIAction.RELEASE:
                    on_mouse_release(algorithm, cursor_log)

            def cursor_cb(x_pos, y_pos):
                cursor_log.pop(0)
                cursor_log.append([int(x_pos), int(y_pos)])

            window.set_keyboard_callback(keyboard_cb)
            window.set_mouse_callback(mouse_cb)
            window.set_cursor_pos_callback(cursor_cb)

            algorithm.on_data_processed(window.show)

            for event in event_stream:
                EventLoop.poll_and_dispatch()
                algorithm.process_data(event)

                if window.should_close():
                    break

                while self.is_paused:
                    EventLoop.poll_and_dispatch(sleep_time_ms=300)
                    if window.should_close():
                        break
                    window.poll_events()
