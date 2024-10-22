import os
from source.data_management.unipen.enum.unipen_enum import UnipenKeywords
from source.data_management.unipen.handlers.base_handler import UnipenHandler

class SingleDocFileHandler(UnipenHandler):
    """This handler manages Unipen provider with a single data file and a data/ folder containing strokes.
    If a or multiple lex file exist, ignore them"""
    config: dict

    def __init__(self, handler_root: str):
        super().__init__(handler_root)

        self.process_doc()

    def process_doc(self):
        """Process the documentation of this Unipen handler, preparing helpers to decipher the stroke data files"""
        #Retrieve single doc file
        files = os.listdir(self.handler_root)
        doc_candidates = [file for file in files if file.endswith(".doc")]
        if len(doc_candidates) != 1:
            raise Exception(f"Provider {self.handler_root}: Expected one doc file, found {doc_candidates}")

        doc_file = doc_candidates[0]

        #Parse doc file in order to find the data required to interpret the stroke
        required_search_values = set([UnipenKeywords.POINTS_PER_SECOND.value, UnipenKeywords.COORD.value])
        opt_search_values = set([
            UnipenKeywords.X_DIM.value, UnipenKeywords.Y_DIM.value,
            UnipenKeywords.X_POINTS_PER_INCH.value, UnipenKeywords.Y_POINTS_PER_INCH.value, 
            UnipenKeywords.X_POINTS_PER_MM.value, UnipenKeywords.Y_POINTS_PER_MM.value, 
        ])

        doc_filepath = os.path.join(self.handler_root, doc_file)
        self.config = {}
        self.config.update(self.search_values_in_file(doc_filepath, required_search_values, throw_on_missing=True))
        self.config.update(self.search_values_in_file(doc_filepath, opt_search_values, throw_on_missing=False))
        

    def get_config_for_datafile(self, datafile_path: str) -> dict:
        """Simple: In single doc mode the configuration is always the same"""
        return self.config