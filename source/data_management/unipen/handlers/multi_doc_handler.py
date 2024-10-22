import os
from source.data_management.unipen.enum.unipen_enum import UnipenKeywords
from source.data_management.unipen.handlers.base_handler import UnipenHandler
from pathlib import Path

class MultiDocFileHandler(UnipenHandler):
    """This handler manages Unipen provider with multiple doc files and a data/ folder containing strokes.
    It parses every config file and assumes that the data folder will be related to the doc files.
    If a or multiple lex file exist, ignore them"""
    configs: dict
    
    def __init__(self, handler_root: str):
        super().__init__(handler_root)

        self.process_doc()

    def process_doc(self):
        """Process the documentation of this Unipen handler, preparing helpers to decipher the stroke data files"""
        files = os.listdir(self.handler_root)
        doc_files = [file for file in files if file.endswith(".doc")]

        self.configs = {}

        for doc_file in doc_files:
            #Parse doc file in order to find the data required to interpret the stroke
            required_search_values = set([UnipenKeywords.POINTS_PER_SECOND.value, UnipenKeywords.COORD.value])
            opt_search_values = set([
                UnipenKeywords.X_DIM.value, UnipenKeywords.Y_DIM.value,
                UnipenKeywords.X_POINTS_PER_INCH.value, UnipenKeywords.Y_POINTS_PER_INCH.value, 
                UnipenKeywords.X_POINTS_PER_MM.value, UnipenKeywords.Y_POINTS_PER_MM.value, 
            ])

            doc_filepath = os.path.join(self.handler_root, doc_file)

            config_file = {}
            
            config_file.update(self.search_values_in_file(doc_filepath, required_search_values, throw_on_missing=True))
            config_file.update(self.search_values_in_file(doc_filepath, opt_search_values, throw_on_missing=False))
            doc_file_key = Path(doc_file).stem
            # some providers have header_{folder}.doc files, it is very easy to remove the header part
            if "header_" in doc_file_key:
                doc_file_key = doc_file_key.removeprefix("header_")
            self.configs[doc_file_key] = config_file

    def get_config_for_datafile(self, datafile_path: str) -> dict:
        """Return the doc file corresponding to the data folder of the data file."""
        data_path = Path(datafile_path)
        data_folder = data_path.parent.name
        try:
            return self.configs[data_folder]
        except KeyError:
            doc_name = data_path.stem
            #For TOS: TOS does not consider the folder to have the same name as the doc file, but a 1;1 datafile - docfile
            return self.configs[doc_name]