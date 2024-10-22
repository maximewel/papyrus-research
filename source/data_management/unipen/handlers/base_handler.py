import os
import re
from source.data_management.unipen.enum.unipen_enum import UnipenKeywords
import numpy as np

class UnipenHandler():
    handler_root: str
    strokes: list[list[int, int, bool]]

    COMMAND_PATTERN = r"^\.(\w*) ?(.*)$"

    #Alignment predefined values
    INCH_TO_MM_RESCALE_FACTOR = 1 / 25.4
    TARGET_PPM = 14
    TARGET_PPS = 200
    #Alignment config
    rescale_strokes_quantiz: bool = True
    resample_strokes: bool = True

    def __init__(self, handler_root: str):
        self.handler_root = handler_root
        self.strokes = []

    def set_alignment_policy(self, rescale_quantization: bool, resample_frequency: bool):
        """Set the stroke alignement policy. Careful: WIll only affect the next process_stroke() call.
        
        Args
        -----
            rescale_quantization: Will align the pixel's spatial quantization to the predefined fixed target PPMM
            resample_frequency: Will align the signal's frequency to the predefined target acquisition frequency
        """
        self.rescale_strokes_quantiz = rescale_quantization
        self.resample_strokes = resample_frequency

    def process_doc(self):
        """Process the documentation of this Unipen handler, preparing helpers to decipher the stroke data files.
        The specific handler prepare its config there"""
        raise NotImplementedError

    def get_config_for_datafile(self, datafile_path: str) -> dict:
        """Get the configuration for the given data file. Specific to handler type."""
        raise NotImplementedError
    
    def search_values_in_file(self, filepath: str, search_keys: set[str], throw_on_missing: bool) -> dict:
        """Helper function.
        Search the given values inside of the file. Return every match in a dictionnary
        The function will either return empty values for missing keys, or throw an exception"""
        result_dict = {}
        found_keys = set()
        
        with open(filepath, "rt") as f:

            for line in f.readlines():
                command_match = re.match(UnipenHandler.COMMAND_PATTERN, line)
                if command_match:
                    if command_match.group(1) in search_keys:
                        key = command_match.group(1)
                        values = command_match.group(2).strip()
                        if " " in values:
                            values = values.split(" ")
                        found_keys.add(key)
                        try:
                            result_dict[key].extend(values)
                        except KeyError:
                            result_dict[key] = values if isinstance(values, list) else [values]

                #Early termination: 
                set_diff = search_keys-found_keys
                if len(set_diff) == 0:
                    break
        
        set_diff = search_keys-found_keys
        if len(set_diff) > 0:
            error = f"Missing key values {set_diff} on file {filepath}"
            if throw_on_missing:
                raise Exception(error)
        
        return result_dict

    def create_strokes(self): 
        """In single doc format, we can open the 'data' folder and expect files to be strokes / folder of files"""
        data_folder = os.path.join(self.handler_root, "data")
        self.scan_data_folder(data_folder)
    
    def scan_data_folder(self, data_folder: str):
        """Scan a data folder in search of stroke files. In case of nested folder, scan nested folders"""
        for filename in os.listdir(data_folder):
            filepath = os.path.join(data_folder, filename)
            if os.path.isdir(filepath):
                self.scan_data_folder(filepath)
            elif os.path.isfile(filepath):
                strokes = self.read_stroke_file(filepath)
                self.strokes.extend(strokes)

    def end_stroke(self, stroke: list, config: dict) -> list:
        """End a stroke and register if it is not empty.
        Return an empty stroke: Either the empty stroke in input or a new empty array"""
        if len(stroke) > 0:
            processed_stroke = self.process_stroke(np.array(stroke), config)
            self.strokes.append(processed_stroke)
            return []
        else:
            return stroke

    def read_stroke_file(self, filepath: str) -> list[tuple[int, int, bool]]:
        """Read a stroke file and return the list as a (x, y, penUp) signal"""
        configuration = self.get_config_for_datafile(filepath)
        coord_config: list = configuration[UnipenKeywords.COORD.value]

        idx, idy = coord_config.index("X"), coord_config.index("Y")

        #Read the line iteratively, line by line. Check if line is an instruction (.INSTRUCTION). If not, try to retrieve coordinates.
        strokes = []
        with open(filepath, "rt") as f:
            #Start at the first PEN instruction
            is_started = False
            pen_down = False
            current_stroke = []
            for line in f:
                line = line.rstrip().strip()
                
                #Gap
                if is_started and not line:
                    current_stroke = self.end_stroke(current_stroke, configuration)
                    continue

                command_match = re.match(UnipenHandler.COMMAND_PATTERN, line)
                if command_match:
                    match command_match.group(1):
                        #the penup signal is encoded in the last value of the stroke
                        case UnipenKeywords.PEN_UP.value:
                            is_started = True
                            pen_down = False
                            if len(current_stroke) > 0:
                                current_stroke[-1][2] = True
                        case UnipenKeywords.PEN_DOWN.value:
                            is_started = True
                            pen_down = True
                            #Well, some providers (anj) put a pen down WITHOUT a pen up, so we have to add the penup signal anyway.
                            if len(current_stroke) > 0:
                                current_stroke[-1][2] = True
                        case _:
                            continue
                else:
                    if not is_started or line.startswith("#"):
                        continue
                    
                    # Not empty line and not command: Assume coordinate line.
                    coords = line.strip().split()

                    x, y = int(coords[idx]), int(coords[idy])
                    if x==0 and y==0:
                        current_stroke = self.end_stroke(current_stroke, configuration)
                    else:
                        # Register points if and only if pen is down.
                        if pen_down:
                            current_stroke.append([x, y, False])

        #Add last current stroke
        self.end_stroke(current_stroke, configuration)

        return strokes

    def resample_by_interpolation(self, signal, input_fs, output_fs):
        """Re-sampling using scypi linear interpolation
        credit to: https://stackoverflow.com/questions/51420923/resampling-a-signal-with-scipy-signal-resample
        Additional credits
        DISCLAIMER: This function is copied from https://github.com/nwhitehead/swmixer/blob/master/swmixer.py, 
        which was released under LGPL. 
        """

        scale = output_fs / input_fs
        # calculate new length of sample. Keep minimum of 1 point
        n = max(round(len(signal) * scale), 1)

        # use linear interpolation
        # endpoint keyword means than linspace doesn't go all the way to 1.0
        # If it did, there are some off-by-one errors
        # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
        # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
        # Both are OK, but since resampling will often involve
        # exact ratios (i.e. for 44100 to 22050 or vice versa)
        # using endpoint=False gets less noise in the resampled sound
        resampled_signal = np.interp(
            np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
            np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
            signal,  # known data points
        )
        return resampled_signal
    
    def quantization_mm(self, config: dict) -> tuple[int, int]:
        """Return the x_per_mm, y_per_mm spatial pixel quantization included in the config. Convert from inch to MM if necessary"""
        x_mm, y_mm = config.get(UnipenKeywords.X_POINTS_PER_MM.value, None), config.get(UnipenKeywords.Y_POINTS_PER_MM.value, None)

        if x_mm is not None:
            x_mm, y_mm = int(x_mm[0]), int(y_mm[0])
        else:
            x_inch, y_inch = config.get(UnipenKeywords.X_POINTS_PER_INCH.value, None), config.get(UnipenKeywords.Y_POINTS_PER_INCH.value, None)
            if x_inch is None:
                raise Exception(f"Found no quantization information in MM or Inch")

            x_mm = int(round(float(x_inch[0]) * UnipenHandler.INCH_TO_MM_RESCALE_FACTOR))
            y_mm = int(round(float(y_inch[0]) * UnipenHandler.INCH_TO_MM_RESCALE_FACTOR))
        
        return x_mm, y_mm
    
    def rescale_stroke(self, stroke: np.ndarray, config: dict) -> None:
        """Rescale the stroke quantization-wize to align the mm value of pixel encoding
        Retrieve the quantization from the config file of the stroke"""
        #Retrieve the quantization of said stroke, always in mm.
        x_mm_quantiz, y_mm_quantiz = self.quantization_mm(config)

        x_quantiz_ratio = x_mm_quantiz / self.TARGET_PPM
        y_quantiz_ratio = y_mm_quantiz / self.TARGET_PPM

        stroke[:, 0] = stroke[:, 0] // x_quantiz_ratio
        stroke[:, 1] = stroke[:, 1] // y_quantiz_ratio

    def resample_stroke(self, stroke: np.ndarray, in_scale: int, out_scale: int) -> np.ndarray:
        """Resample a stroke by resampling every sub-stroke, thus respecting the penup signals"""
        #Cut the signal into sub-signal separated by the penup instruction
        substrokes = []
        current_substroke = []

        for (x, y, penup) in stroke:
            current_substroke.append([x, y])
            if penup:
                substrokes.append(np.array(current_substroke))
                current_substroke = []

        if len(current_substroke) > 0:
            substrokes.append(np.array(current_substroke))
            
        #Resample x,y values
        resampled_substrokes = []
        for substroke in substrokes:
            x_resampled = np.rint(self.resample_by_interpolation(substroke[:, 0], in_scale, out_scale)).astype(int)
            y_resampled = np.rint(self.resample_by_interpolation(substroke[:, 1], in_scale, out_scale)).astype(int)
            penup_resampled = np.full_like(x_resampled, fill_value=False)
            penup_resampled[-1] = True

            resampled_substroke = np.concatenate(
                (x_resampled[:, np.newaxis], y_resampled[:, np.newaxis], penup_resampled[:, np.newaxis]), 
                axis=1
            )

            resampled_substrokes.append(resampled_substroke)

        #Re-construct the stroke
        final_stroke = np.concatenate(resampled_substrokes, axis=0, dtype=np.int32)
        return final_stroke

    def process_stroke(self, stroke: list[tuple[int, int, bool]], config: dict):
        """Apply post processing to the strokes
        Config: Configuration of the stroke, especially containing spatial info (DPI, PPI, etc) and spatial info (Points per seconds)"""
        align_strokes = True
        start_padding = 5

        #Y axis is inversed between plotting and signal
        max_y = max(stroke[:, 1])
        stroke[:, 1] = max_y - stroke[:, 1]

        #Alignement / padding
        if align_strokes:
            min_x = min(stroke[:, 0])
            stroke[:, 0] -= (min_x - start_padding)

            min_y = min(stroke[:, 1])
            stroke[:, 1] -= (min_y - start_padding)

        # Frequency alignement
        if self.resample_strokes:
            signal_freq = int(config[UnipenKeywords.POINTS_PER_SECOND.value][0])
            target_pps = self.TARGET_PPS #100PPS, or 10MS
            if signal_freq != target_pps:
                stroke = self.resample_stroke(stroke, signal_freq, target_pps)
        
        #Quantization alignement
        if self.rescale_strokes_quantiz:
            self.rescale_stroke(stroke, config)

        return stroke