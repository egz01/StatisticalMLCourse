import tkinter as tk
import pickle
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

def gaussian_kernel(size, sigma):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))
    #return g / g.sum()
    return g

def calc_Y_matrix_form(W, X):
    Z = np.dot(W, X.T)
    Z -= np.max(Z, axis=0)
    normalizer = np.sum(np.exp(Z), axis=0)
    normalizer = np.vstack(normalizer)
    Y = (1/normalizer)*np.exp(Z).T
    return Y

def calculate_predictions(W, X):
    Y = calc_Y_matrix_form(W, X)
    return np.argmax(Y, axis=1)

def preprocess(img: np.ndarray, scaler: StandardScaler):
    img = img.astype(np.float64)
    img = img.reshape((784,))
    img = np.concatenate((img, np.ones(1)))
    img = img.reshape((1, -1))
    img = scaler.transform(img)
    img[:, -1] = 1
    return img

class MnistExperiment:
    def __init__(self, root, weights, scaler):
        self.root = root
        self.root.title("MNIST Experiment")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        # create a canvas to draw digits on, set size to 28x28
        # add grid to canvas to help user draw
        self.canvas = tk.Canvas(self.root, width=28*10, height=28*10, bg="white")
        self.canvas.place(x=400-28*5, y=300-28*5)

        # make canvas drawable by mouse click and drag:
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<B3-Motion>", self.erase)
        self.canvas.bind("<Button-1>", self.draw)
        self.canvas.bind("<Button-3>", self.erase)

        # add label to canvas to display result
        self.result_label = tk.Label(self.root, text="Identified digit: ", font=("Arial", 20))
        self.result_label.place(x=100, y=100)
        
        self.reset_button = tk.Button(self.root, text="Reset", font=("Arial", 20), command=self.reset)
        self.reset_button.place(x=100, y=500)

        self.identify_button = tk.Button(self.root, text="Identify", font=("Arial", 20), command=self.identify)
        self.identify_button.place(x=600, y=500)

        self.show_analysis = False
        self.show_analysis_cb = tk.Checkbutton(self.root, text="Show analysis", font=("Arial", 14), command=lambda: setattr(self, "show_analysis", not self.show_analysis))
        self.show_analysis_cb.place(x=600, y=550)

        self.reset()
        # make GUI non blocking
        self.root.mainloop()

    def draw(self, event):
        # each grid cell touched by a clicked mouse is colored black
        # get coordinates of mouse click
        x = event.x
        y = event.y

        # get grid cell coordinates
        x = x // 10
        y = y // 10

        # color grid cell black
        if 0 < x < 28 and 0 < y < 28:
            self.image_array[y, x] = 255
            self.canvas.create_rectangle(x*10, y*10, x*10+10, y*10+10, fill="black")

    def erase(self, event):
        # each grid cell touched by a clicked mouse is colored black
        # get coordinates of mouse click
        x = event.x
        y = event.y

        # get grid cell coordinates
        x = x // 10
        y = y // 10

        # color grid cell black
        if 0 < x < 28 and 0 < y < 28:
            self.image_array[y, x] = 0
            self.canvas.create_rectangle(x*10, y*10, x*10+10, y*10+10, fill="white", outline="gray", width=1)

    def reset(self):
        self.canvas.delete("all")
        self.image_array = np.zeros((28, 28))
        self.result_label.config(text="Identified digit: ")
        for i in range(1, 28):
            self.canvas.create_line(i*10, 0, i*10, 28*10, fill="gray")
            self.canvas.create_line(0, i*10, 28*10, i*10, fill="gray")
        
    def identify(self):
        # add gaussian noise and blur to make the digit similar to the training set
        img = self.image_array.astype(np.float64).copy()
        img = signal.convolve2d(img, gaussian_kernel(5, 1), mode="same")
        img = img*255/np.max(img)
        img = np.round(img)
        img = np.clip(img, 0, 255)
        self.altered_image = img.copy()

        x = preprocess(img, scaler)

        if self.show_analysis:
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(img, cmap="gray")
            axs[1].imshow(x[0][:-1].reshape((28, 28)), cmap='gray')
            plt.show()

        predictions = calculate_predictions(weights, x)
        self.result_label.config(text=f"Identified digit: {predictions[0]}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", type=str, default="weights/weights.pkl", help="Path to weights file")
    parser.add_argument("-s", "--scaler", type=str, default="weights/scaler.pkl", help="Path to scaler file")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    with open(args.weights, "rb") as f:
        weights = pickle.load(f)
    with open(args.scaler, "rb") as f:
        scaler = pickle.load(f)

    g = MnistExperiment(tk.Tk(), weights, scaler)

    plt.close()