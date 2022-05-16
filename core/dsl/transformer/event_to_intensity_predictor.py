import numpy as np
from scipy.ndimage import gaussian_filter

from core.constants.colors import WHITE
from core.dsl.transformer.module import Transformer


class AsymptoticIntensityPredictor(Transformer):

    def __init__(self,
                 gaussian_filter_sigma=1.0,
                 intensity_decay=0.2,
                 intensity_impedance=1.0):
        super().__init__()

        self.sigma = gaussian_filter_sigma
        self.intensity_decay = intensity_decay
        self.impedance = intensity_impedance

        self.screen_buffer = None
        self.intensity = None

    def late_init(self, height, width, **kwargs):
        self.screen_buffer = np.full([height, width, 3], WHITE, dtype=np.ubyte)
        self.intensity = np.zeros((height, width), dtype=np.float64)

    def process_data(self, events, **kwargs):

        # Find positive/negative event indicis.
        delta_up = np.argwhere(events['p'] == 1)
        delta_down = np.argwhere(events['p'] == 0)

        # Calculate decaying value for individual pixels. Pixels with higher
        # intensities, will decay more than pixels with lower intensities.
        decay = self.intensity_decay * self.intensity
        # Only apply decay on pixels with no current event.
        # Therefore, set decay to zero for pixels with events.
        decay[events['y'], events['x']] = 0
        # Finally, subtract pixel intensities. Subtract zero from pixels with events, leaving them unaffected.
        self.intensity -= decay

        # Save pixel intensities with current events temporarily array. We want to apply gaussian filtering
        # on the picture to remove ghosting, but also want to revert the filtering for these pixels.
        temp = self.intensity[events['y'], events['x']]
        # Apply the gaussian on the matrix, essentially blurring the picture.
        self.intensity = gaussian_filter(self.intensity, sigma=self.sigma)
        # Revert changes for event pixels.
        self.intensity[events['y'], events['x']] = temp

        # Find pixel coordinates with their respective polarity.
        y_pos, x_pos = events['y'][delta_up], events['x'][delta_up]
        y_neg, x_neg = events['y'][delta_down], events['x'][delta_down]

        '''
        Calculating new intensity for pixels with positive events.
        Intensities are values between 1 (white) and -1 (black). Since offsets are unknown information
        from the perspective of event cameras, we assume pixels start right in the middle (grey), i.e. we dont,
        make any bias assumption that it starts at black. If a pixel is excited by a positive event,
        the intensity increases a bit, and subsequent excitations will continue increasing it, but at
        weaker magnitudes at each step. Pixels require infinite positive events to go pure white,
        and likewise infinite to go pitch black. The intensity is therefore asymptotic since it may get close,
        but never exactly at the boundary. When pixels are very white, they get more sensitive to negative
        events and vice versa. This schemes ensures number overflows cannot occur, and a high resilience to noice.
        The formula:
        
        let i = intensity
        let j = impedance
        
        i_new = j*(i_old + (1 - i_old)/2)
        
        The impedance are simply a number between 0 and 1 inclusive which purpose is 
        to slow down the changes in intensities.
        '''

        self.intensity[y_pos, x_pos] += self.impedance * ((1 - self.intensity[y_pos, x_pos]) / 2)

        '''
        Same, but for negative events, which entail decrease in intensity value.
        Now addition is flipped to subtraction, and vice versa.
        
        let i = intensity
        let j = impedance
        
        i_new = j*(i_old - (1 + i_old)/2)
        
        '''
        self.intensity[y_neg, x_neg] -= self.impedance * ((1 + self.intensity[y_neg, x_neg]) / 2)

        '''
        Now we calculate the greyscale for each pixel. Remember that intensities are bounded between -1 and 1.
        We add intensities by one to get a positive range, then multiply by greyscale-space value divided by 2
        to scale the number between 0 and max 255.  
        '''

        greyscale_values = ((self.intensity + 1) * (225 / 2)).astype(np.ubyte)

        # Send to screen.
        self.screen_buffer[:, :] = np.repeat(greyscale_values[:, :, np.newaxis], 3, axis=2)

        # Done.
        self.callback(self.screen_buffer, **kwargs)
