from source.data_management.common.handwritting_dataset import HandWrittingDataset
import numpy as np

from source.logging.log import logger, LogChannels

class StrokedHandwrittingDataset(HandWrittingDataset):
    #Some constants used during pre-processing of the signals before creating the images
    PADDING = 3
    MAX_PIX_PENUP_THRESHOLD = 100
    OUTLIER_STD_THRESH = 2

    # Use only in test - if true, save the X biggest images to local file. Keep at false.
    # Used for checking that the X biggest images are still somehow readable after passing through the 
    # Resize-to-mean function
    SAVE_RESIZES_TO_FILE = False
    SAVE_IMAGES_NUMBER = 4

    separate_strokes: bool

    def __init__(self, patches_dim, separate_strokes: bool = True, normalize_pixel_values = True, normalize_coordinate_sequences = True, 
                 window_size = None, lstm_forecast_length = None, samples_to_take: int|float = None):
        self.separate_strokes = separate_strokes
        
        super().__init__(patches_dim, normalize_pixel_values, normalize_coordinate_sequences, window_size, lstm_forecast_length, samples_to_take)

    def apply_all_preprocess_to_signals(self):
        """
        Apply all common pre-processes to the signals. Must be called before creating images.
        """
        logger.log(LogChannels.DATA, f"Processing data. Initial Number of signals: {len(self.signals)}")

        #Apply artificual cuts to signal
        logger.log(LogChannels.DATA, f"Separating strokes...")
        self.apply_stroke_separation_to_signals()
        logger.log(LogChannels.DATA, f"Strokes done. Number of signals: {len(self.signals)}")

        logger.log(LogChannels.DATA, f"Applying windows...")
        self.apply_window_to_signals()
        logger.log(LogChannels.DATA, f"Windows done. Number of signals: {len(self.signals)}")

        #Since signal is cut, align it
        logger.log(LogChannels.DATA, f"Aligning all signals...")
        self.align_all_signal()
        logger.log(LogChannels.DATA, f"Signals aligned.")

        #Apply further dataset processing to reduce the size of the maximum image
        logger.log(LogChannels.DATA, f"Checking suspsicious PenUp signals...")
        self.verify_penup_signals()
        logger.log(LogChannels.DATA, f"Penup verification done. Number of signals: {len(self.signals)}")

        logger.log(LogChannels.DATA, f"Removing all outliers...")
        self.remove_outlier_images_and_resize()
        logger.log(LogChannels.DATA, f"Outliers removed. Number of signals: {len(self.signals)}")

        #Apply a last align
        logger.log(LogChannels.DATA, f"Aligning all signals...")
        self.align_all_signal()
        logger.log(LogChannels.DATA, f"Signals aligned.")

    def apply_stroke_separation_to_signals(self) -> list[list]:
        """Change the signals to substrokes if requested / necessary"""
        if self.separate_strokes:
            signals_as_substrokes = []
            for signal in self.signals:
                substroke_signals = self.separate_signal_in_strokes(signal)
                signals_as_substrokes.extend(substroke_signals)
            self.signals = signals_as_substrokes

    @classmethod
    def separate_signal_in_strokes(cls, signal: list) -> list[list]:
        """Provided an online signal, return a list of its substrokes separated by PENUP
        
        args
        -----
            signal: list - The signal as (x,y,penup) array
            align_substrokes: bool - Whether to align the substrokes at the left
            h_padding: 
        """
        substrokes = []

        start_idx = 0
        end_idx = 0

        #Cut the stroke each time the pen is up to obtain substrokes
        for x, y, penup in signal:
            end_idx += 1
            if penup:
                substroke = signal[start_idx:end_idx]                
                substrokes.append(substroke)
                start_idx = end_idx

        return substrokes
    
    def align_all_signal(self):
        """
        Substrokes can be aligned left and up by removing their minimum X,Y value.
        Usefull when we cut the signal.
        """
        for signal in self.signals:
            # If a substroke starts at x=40, it can be re-placed at 0. 
            # This is be done via substracting the min X value to each value of the signal
            min_x = min(signal[:, 0])
            signal[:, 0] -= (min_x - self.PADDING)
            
            # Similar for Y values
            min_y = min(signal[:, 1])
            signal[:, 1] -= (min_y - self.PADDING)
    
    def apply_window_to_signals(self):
        """Cut the given list of signals into sub-signals, regroup them into a single list"""
        #If window size, each signal has to be passed through the apply window function that splits signals that are too long
        if self.window_size:
            window_signals = []
            for raw_signal in self.signals:
                window_signals.extend(self.apply_window_to_signal(raw_signal))
            self.signals = window_signals

    def apply_window_to_signal(self, signal: list[tuple]):
        """Cut the signal into sub-signals
        Args
        -----
            signal: list[tuple] - The signal, as a list of tuples (x,y,penUp)
        """
        if len(signal) <= self.window_size:
            return [signal]
        
        signal_numbers = int(np.ceil(len(signal)/self.window_size))
        
        logger.log(LogChannels.DATA, f"Cutting signal of len {len(signal)} in {signal_numbers}")

        cut_signals = []

        for i in range(signal_numbers):
            cut_signals.append(signal[i*self.window_size : (i+1)*self.window_size])

        logger.log(LogChannels.DATA, f"returning {len(cut_signals)} signals of len {[len(sig) for sig in cut_signals]}")

        return cut_signals

    def remove_outlier_images_and_resize(self):
        """
        Remove outlier images in term of image width and height, in order to not waste high processing power as the max length
        image dictates the dimensions of the image tensor and padding
        """
        images_widths, images_heigths = [], []
        for signal in self.signals:
            #Get X, Y amplitudes
            min_X, max_X = np.min(signal[:, 0]), np.max(signal[:,0])
            min_Y, max_Y = np.min(signal[:, 1]), np.max(signal[:,1])
            image_w, image_h = max_X - min_X, max_Y - min_Y
            images_widths.append(image_w)
            images_heigths.append(image_h)

        images_widths_mean, images_widths_std = np.mean(images_widths), np.std(images_widths)
        images_heigths_mean, images_heigths_std = np.mean(images_heigths), np.std(images_heigths)

        W_thresh, H_thresh = int(images_widths_mean + self.OUTLIER_STD_THRESH*images_widths_std), int(images_heigths_mean + self.OUTLIER_STD_THRESH*images_heigths_std)

        #Remove worst outliers from dataset by eliminating them from all three lists in order to re-use the calculated stats
        outlier_indexes = [i for i, _ in enumerate(self.signals) if images_widths[i] > W_thresh or images_heigths[i] > H_thresh]
        
        self.signals = [signal for i, signal in enumerate(self.signals) if i not in outlier_indexes]
        images_widths = [width for i, width in enumerate(images_widths) if i not in outlier_indexes]
        images_heigths = [heigth for i, heigth in enumerate(images_heigths) if i not in outlier_indexes]
        
        #Only if activated, flag the biggest N images
        if self.SAVE_RESIZES_TO_FILE:
            image_sizes = np.array([w * h for (w, h) in zip(images_widths, images_heigths)])
            largest_indices = np.argsort(image_sizes)[-self.SAVE_IMAGES_NUMBER:][::-1]
            images_before_after_resize = []

        #Resize images too big, preserving aspect ratio
        for i in range(len(self.signals)):
            image_w = images_widths[i]
            image_h = images_heigths[i]

            ratio_w_to_mean = image_w / (images_widths_mean + images_widths_std)
            ratio_h_to_mean = image_h / (images_heigths_mean + images_heigths_std)

            if ratio_w_to_mean > 1 or ratio_h_to_mean > 1:
                resize_factor = 1 / max(ratio_w_to_mean, ratio_h_to_mean)
                original_image = self.signals[i].copy()
                self.signals[i] = np.round(original_image * (resize_factor, resize_factor, 1)).astype(int)
            
            if self.SAVE_RESIZES_TO_FILE and i in largest_indices:
                images_before_after_resize.append((original_image, self.signals[i]))
        
        if self.SAVE_RESIZES_TO_FILE:
            self.plot_multiple_before_after(images_before_after_resize)

    def plot_multiple_before_after(self, signals_list: list[tuple[list, list]], save_path='./multiple_before_after_resize.png'):
        """
            If necessary, plot multiple before/after image pairs stored in the image list.
            Very quick method only used for test.
        """
        #Local imports, never used except during this particular test
        import matplotlib.pyplot as plt
        from source.model.blocks.constants.sequence_to_image import ImageHelper

        num_images = len(signals_list)

        # Create a figure with enough space for all rows and a fixed number of columns (2 * N)
        plt.figure(figsize=(15, 4 * num_images))

        plt.suptitle("Image before-after the resize", fontsize=16)

        for i, (signalBeforeResize, signalAfterResize) in enumerate(signals_list):
            imageBeforeResize, imageAfterResize = ImageHelper.create_image(signalBeforeResize), ImageHelper.create_image(signalAfterResize)
            
            # Plot the image before resize on the left (column 1)
            plt.subplot(num_images, 2, 2 * i + 1)
            plt.xlabel('Width (pixels)')
            plt.ylabel('Height (pixels)')
            plt.imshow(imageBeforeResize, cmap='gray', extent=[0, imageBeforeResize.shape[1], 0, imageBeforeResize.shape[0]])
            plt.title(f"Image before resize ({i+1})")

            # Plot the image after resize on the right (column 2)
            plt.subplot(num_images, 2, 2 * i + 2) 
            plt.xlabel('Width (pixels)')
            plt.ylabel('Height (pixels)')
            plt.imshow(imageAfterResize, cmap='gray', extent=[0, imageAfterResize.shape[1], 0, imageAfterResize.shape[0]])
            plt.title(f"Image after resize ({i+1})")

        # Adjust layout and save the figure to a file
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path)

        # Display the plot
        plt.show()

    def verify_penup_signals(self):
        """
        Some PenUp instructions are not given correctly, resulting in very wide images corresponding to pen movements.
        Try to remove the worse outliers by artifially creating penups (Cutting) the data where the jump between two 
        points is above a given threshold
        """
        checked_signals = []
        for signal in self.signals:
            checked_signals.extend(self.verify_penup_signal(signal))
        self.signals = checked_signals
    
    def verify_penup_signal(self, signal: list) -> list[list]:
        """
        Cut a signal into subsignals as long as the euclidian distance between two points is above a threshold
        """        
        if len(signal) < 2:
            return [signal]

        #Compute euclidian distances from each point to another
        deltas = signal[1:] - signal[:-1]
        distances = np.linalg.norm(deltas, axis=1)
        
        #If dist > threshold, this is suspicious as the pen doesn't have infinite speed. Cut these links.
        split_indices = np.where(distances > self.MAX_PIX_PENUP_THRESHOLD)[0] + 1
        segments = np.split(signal, split_indices)
        return segments