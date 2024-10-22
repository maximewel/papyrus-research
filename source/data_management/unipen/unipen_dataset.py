from source.logging.log import logger, LogChannels

from source.data_management.unipen.handlers.handler_builder import UnipenHandlerBuilder
from source.data_management.common.stroked_handwriting_dataset import StrokedHandwrittingDataset

class UnipenDataset(StrokedHandwrittingDataset):
    unipen_root: str

    def __init__(self, unipen_root: str, patches_dim, strokemode: bool = True, normalize_pixel_values = True, 
                 normalize_coordinate_sequences = True, 
                 window_size = None, lstm_forecast_length = None,
                 samples_to_take: int|float = None):
        self.unipen_root = unipen_root
        
        super().__init__(patches_dim, strokemode, normalize_pixel_values, normalize_coordinate_sequences, window_size, lstm_forecast_length, samples_to_take)
    
    #Override
    def _load_data(self):
        """Load all the signals and images"""
        self.load_raw_data()

    def load_raw_data(self):
        """ 
        Load all the Unipen raw data
        the Unipen data is into multiple formats. Use a Handler builder tobuild handlers correpsonding to the
        data fromats to obtain all strokes.
        """
        #Get all the UNIPEN handlers from the handler builder
        unipen_handler_builder = UnipenHandlerBuilder(self.unipen_root)
        handlers = unipen_handler_builder.build_handlers()
        logger.log(LogChannels.DATA, f"Unipen - built {len(handlers)} handlers")

        #Ask each handler to retrieve its data, retrieve it internally
        logger.log(LogChannels.DATA, f"Unipen - Creating strokes...")
        total_len = len(handlers)
        ind = 1
        for handler in handlers:
            logger.log(LogChannels.DATA, f"Processing handler {ind}/{total_len}")
            handler.create_strokes()
            self.signals.extend(handler.strokes)
            ind += 1
        logger.log(LogChannels.DATA, f"Stroke creation done")

        self.apply_all_preprocess_to_signals()

        if self.samples_to_take is not None:
            self.take_samples_of_dataset(self.samples_to_take)

        self.build_images()