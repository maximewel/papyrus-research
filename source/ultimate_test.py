import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from source.model.blocks.constants.sequence_to_image import ImageHelper
from source.model.blocks.constants.files import *
from source.data_management.common.handwritting_dataset import HandWrittingDataset
from source.model.hw_model import HwTransformer
from source.model.blocks.constants.files import *
from source.model.blocks.constants.device_helper import device
from torch.nn.utils.rnn import pack_sequence
from source.model.blocks.constants.datasets_library import *
from dtw import *
import numpy as np
import torch
from tkinter import *
from PIL import Image, ImageTk
from time import sleep

MODEL_NAME = "best_m_20_epochs"
PATCHES_DIM = (16, 16)
target_image_shape = (96, 96)

STOP_INK = True

sample_rate = 100

class DrawingApp:

    MULT_EFFECT = 2
    CANVAS_BASE_SIZE = 96

    model: HwTransformer

    def create_prediction_signal(self, image, first_coordinate, stop_len, stop_ink) -> list:
        #Obtain patch and padding of image. Assume no paddding.
        patched_image, padding = HandWrittingDataset.images_to_tensor([image], PATCHES_DIM, target_image_shape)
        current_signal = torch.Tensor([first_coordinate])

        patched_image, padding, current_signal = patched_image.to(device), padding.to(device), current_signal.to(device)
        working_signal = current_signal[:1]

        i = 1
        stop_signal = False
        while not stop_signal:
            #Generate next point
            res = self.model.forward(patched_image, padding, pack_sequence(working_signal.unsqueeze(0)))
            res = res.detach().round()

            #Add it to working signal
            working_signal = torch.vstack([working_signal, res])

            # Check if we should stop by ink method
            if STOP_INK:
                diff = working_signal[1:] - working_signal[:-1]
                distances = torch.sqrt(torch.sum(diff**2, dim=1))
                sum_of_pix = distances.sum()
                print(f"pix{sum_of_pix} / {stop_ink}")
                stop_signal = (sum_of_pix >= stop_ink) or (len(working_signal) >= stop_len)
                if(stop_signal):
                    working_signal = working_signal[:-1]
            else:
                stop_signal = (len(working_signal) >= stop_len)

            i += 1

        return working_signal.cpu().numpy()

    def create_drawing_canvas(self):
        # Add a title label above the canvas
        self.title_label = tk.Label(root, text="Draw", font=("Arial", 14))
        self.title_label.grid(row=0, column=0, pady=(5, 5))

        # Create a canvas with padding
        self.canvas_frame = tk.Frame(root, padx=10, pady=10, bg="gray")
        self.canvas_frame.grid(row=1, column=0)

        self.canvas = tk.Canvas(self.canvas_frame, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()
    
    def create_canvas_compute(self):
        # Add a title label above the canvas
        self.title_label_showing = tk.Label(root, text="Sent", font=("Arial", 14))
        self.title_label_showing.grid(row=0, column=1, pady=(5, 5))

        # Create a canvas with padding
        self.canvas_compute_frame = tk.Frame(root, padx=10, pady=10, bg="gray")
        self.canvas_compute_frame.grid(row=1, column=1)

        self.canvas_compute = tk.Canvas(self.canvas_compute_frame, width=self.CANVAS_BASE_SIZE, height=self.CANVAS_BASE_SIZE, bg="white")
        self.canvas_compute.pack()

    def create_showing_canvas(self):
        # Add a title label above the canvas
        self.title_label_showing = tk.Label(root, text="Computed", font=("Arial", 14))
        self.title_label_showing.grid(row=0, column=2, pady=(5, 5))

        # Create a canvas with padding
        self.canvas_frame_showing = tk.Frame(root, padx=10, pady=10, bg="gray")
        self.canvas_frame_showing.grid(row=1, column=2)

        self.canvas_showing = tk.Canvas(self.canvas_frame_showing, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas_showing.pack()

    def __init__(self, root):
        self.root = root
        self.root.title("Drawing App")

        #Init model
        model_path = os.path.join('.', SOURCE_FILENAME, MODEL_FOLDER, TRANSFORMER_FOLDER, MODEL_NAME, MODEL_FILENAME)
        print(f"Loading model from: {model_path}")
        self.model = torch.load(model_path)
        self.model.eval()

        # Canvas dimensions
        self.canvas_size = self.MULT_EFFECT * self.CANVAS_BASE_SIZE

        # Create a canvas
        self.create_drawing_canvas()

        #Create second canvas for compute
        self.create_canvas_compute()

        #Create third canvas for showing
        self.create_showing_canvas()


        # Buttons
        self.restart_button = tk.Button(root, text="Restart", command=self.restart)
        self.restart_button.grid(row=2, column=0, pady=10)

        self.compute_button = tk.Button(root, text="Compute", command=self.compute)
        self.compute_button.grid(row=2, column=1, pady=10)

        self.play_button = tk.Button(root, text="Play", command=self.play)
        self.play_button.grid(row=2, column=2, pady=10)
        self.play_button.config(state=DISABLED)

        # Variables for drawing
        self.drawing = False
        self.coordinates = []
        self.image = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        self.canvas.bind("<B1-Motion>", self.draw)

    def start_draw(self, event):
        self.drawing = True
        self.coordinates = []  # Reset coordinates for a new stroke
        self.record_coordinates()

    def stop_draw(self, event):
        self.drawing = False

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_oval(x, y, x+1, y+1, fill="black", outline="black")
            self.image[y, x] = 1  # Update the numpy array

    def get_mouse_position(self):
        # Get the mouse position relative to the root window
        root_x, root_y = self.root.winfo_pointerx(), self.root.winfo_pointery()
        # Get the canvas position within the root window
        canvas_x, canvas_y = self.canvas.winfo_rootx(), self.canvas.winfo_rooty()
        # Calculate mouse position relative to the canvas
        mouse_x, mouse_y = root_x - canvas_x, root_y - canvas_y
        return mouse_x, mouse_y
    
    def record_coordinates(self):
        if self.drawing:
            # Record current coordinates
            self.coordinates.append(self.get_mouse_position())
            self.root.after(sample_rate, lambda: self.record_coordinates())

    def restart(self):
        self.canvas.delete("all")
        self.canvas_showing.delete("all")
        self.canvas_compute.delete("all")
        self.coordinates = []
        self.image.fill(0)
        self.compute_button.config(state=ACTIVE)

    def compute(self):
        coordinates = np.array(self.coordinates) / self.MULT_EFFECT

        img = ImageHelper.create_image(torch.nn.functional.pad(torch.tensor(coordinates, dtype=int), (0, 1)).numpy(), (93, 93))
        
        img_tk_Format = Image.fromarray((img * 255).astype(np.uint8))
        self.img_tk = ImageTk.PhotoImage(img_tk_Format)
        self.canvas_compute.create_image(0, 0, anchor=NW, image=self.img_tk)

        #Get first coordinate
        x,y = coordinates[0]
        x_scaled, y_scaled = x * self.MULT_EFFECT, y * self.MULT_EFFECT
        radius = 2
        self.canvas.create_oval(
            x_scaled - radius, y_scaled - radius, x_scaled + radius, y_scaled + radius, 
            fill="red", outline="red"
        )
        self.root.update()

        #Obtain coordinate vector from image
        self.signal = self.create_prediction_signal(img, (x,y), 1.5*len(coordinates), np.sum(img))

        self.compute_button.config(state=DISABLED)
        self.play_button.config(state=ACTIVE)

    def play(self):
        for i in range(len(self.signal) - 1):
            x1, y1 = self.signal[i]
            x2, y2 = self.signal[i + 1]
            self.canvas_showing.create_line(x1*self.MULT_EFFECT, y1*self.MULT_EFFECT, x2*self.MULT_EFFECT, y2*self.MULT_EFFECT, fill="black", width=2)
            self.root.update()
            sleep(sample_rate / 1000)

        self.play_button.config(state=DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()