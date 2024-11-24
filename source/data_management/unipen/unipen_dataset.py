from source.logging.log import logger, LogChannels

from source.data_management.unipen.handlers.handler_builder import UnipenHandlerBuilder
from source.data_management.common.stroke_handwriting_dataset import StrokeHandwrittingDataset

class UnipenDataset(StrokeHandwrittingDataset):
    unipen_root: str

    def __init__(self, unipen_root: str, separate_strokes: bool = True, image_max_shape: tuple[int, int] = None, window_size: int = None, save_to_file: bool = False):
        self.unipen_root = unipen_root
        
        super().__init__(unipen_root, separate_strokes, image_max_shape, window_size, save_to_file)
    
    def _load_raw_data(self):
        """ 
        Load all the Unipen raw data
        the Unipen data is into multiple formats. Use a Handler builder tobuild handlers correpsonding to the
        data fromats to obtain all strokes.
        """
        self.signals = []

        #Get all the UNIPEN handlers from the handler builder
        unipen_handler_builder = UnipenHandlerBuilder(self.unipen_root)
        handlers = unipen_handler_builder.build_handlers()
        logger.log(LogChannels.DATA, f"Unipen - built {len(handlers)} handlers")

        #Ask each handler to retrieve its data, retrieve it internally
        logger.log(LogChannels.DATA, f"Unipen - Retrieving original signals...")
        total_len = len(handlers)
        ind = 1
        for handler in handlers:
            logger.log(LogChannels.DATA, f"Processing handler {ind}/{total_len}")
            handler.create_strokes()
            self.signals.extend(handler.strokes)
            ind += 1
        logger.log(LogChannels.DATA, f"original Signals loaded")