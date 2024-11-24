import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from source.data_management.brush.brush_dataset import BrushDataset
from source.data_management.unipen.unipen_dataset import UnipenDataset
from source.data_management.common.handwritting_dataset import HandWrittingDataset
from source.model.blocks.constants.files import *
from source.logging.log import logger, LogChannels

def prepare_unipen():
    print(F"Loading brush, saving on file")
    datasource = UnipenDataset(unipen_root=UNIPEN_ROOT, separate_strokes=True, save_to_file=True, image_max_shape=(100,100))
    print(F"Loading Done, number of signals: {len(datasource.signals)}")


    print(f"Verification - this version should load immediatly")
    datasource_from_file = UnipenDataset(unipen_root=UNIPEN_ROOT, separate_strokes=True, image_max_shape=(100,100))
    print(F"Loading Done, number of signals: {len(datasource_from_file.signals)}")
    
if __name__ == "__main__":
    logger.add_log_channel(LogChannels.DATA)

    prepare_unipen()