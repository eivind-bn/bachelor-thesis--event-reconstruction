import metavision_core.utils.samples as samples

from core.transformer.frames_by_timestamps import FramesByTimestamps

samples.get_all_samples('samples/events/')

def event(filename, suffix='raw'):
    return f'samples/events/{filename}.{suffix}'

def mp4(filename):
    return f'samples/mp4/{filename}.mp4'

def recon(filename):
    return f'samples/recon/{filename}.mp4'

def decon(filename):
    return f'samples/decon/{filename}.dat'

from core.transformer.frames_by_event_batches import EventBatchToFrames
from core.sink.window import Window
from core.source.events_reader import EventReader

EventReader(decon('formula1'), delta_t=1e4) >> FramesByTimestamps() >> Window('image')
# EventReader(decon('formula1'), delta_t=1e4).on_data_processed(AsyncFrameGen().on_data_processed(Window('image')))