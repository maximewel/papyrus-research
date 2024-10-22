import os
from source.data_management.unipen.enum.unipen_enum import UnipenKeywords
from source.data_management.unipen.handlers.base_handler import UnipenHandler
from pathlib import Path

class NoDocFileHandler(UnipenHandler):
    """This handler manages Unipen provider with no doc files and a data/ folder containing strokes.
    In this configuration, it is assumed that each data file must contain the documentation at its start. As the search in file function returns
    when values are found, it means minimal overhead (2IO operations instead of 1)
    If a or multiple lex file exist, ignore them"""
    configs: dict
    
    def __init__(self, handler_root: str):
        super().__init__(handler_root)

        self.process_doc()

    def process_doc(self):
        """Process the documentation of this Unipen handler, assuming the data files are the documentation"""
        data_path = os.path.join(self.handler_root, "data")
        files = os.listdir(data_path)

        self.configs = {}

        for file in files:
            #Parse doc file in order to find the data required to interpret the stroke
            required_search_values = set([UnipenKeywords.POINTS_PER_SECOND.value, UnipenKeywords.COORD.value])
            opt_search_values = set([
                UnipenKeywords.X_DIM.value, UnipenKeywords.Y_DIM.value,
                UnipenKeywords.X_POINTS_PER_INCH.value, UnipenKeywords.Y_POINTS_PER_INCH.value, 
                UnipenKeywords.X_POINTS_PER_MM.value, UnipenKeywords.Y_POINTS_PER_MM.value, 
            ])

            doc_filepath = os.path.join(data_path, file)
            if os.path.isdir(doc_filepath):
                child_files = [os.path.join(file, child_file) for child_file in os.listdir(doc_filepath)]
                files.extend(child_files)
                continue

            config_file = {}
            
            config_file.update(self.search_values_in_file(doc_filepath, required_search_values, throw_on_missing=True))
            config_file.update(self.search_values_in_file(doc_filepath, opt_search_values, throw_on_missing=False))
            doc_file_key = Path(doc_filepath).stem
            # some providers have header_{folder}.doc files, it is very easy to remove the header part
            if "header_" in doc_file_key:
                doc_file_key = doc_file_key.removeprefix("header_")
            self.configs[doc_file_key] = config_file

    def get_config_for_datafile(self, datafile_path: str) -> dict:
        """Return the doc file corresponding to the data folder of the data file."""
        doc_file_key = Path(datafile_path).stem
        return self.configs[doc_file_key]