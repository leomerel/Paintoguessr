import tkinter as tk
import tkinter.ttk as ttk
import numpy as np

import imageHandling
import plot
import time

class Paint:
    BACKGROUND_COLOR = "#333333"

    def __init__(self, cnn):
        self.cnn = cnn
        self.start = time.time_ns()

        self.window = tk.Tk()
        self.window.title("Paintoguessr")
        self.window.configure(bg=self.BACKGROUND_COLOR)

        self.lbl_title = tk.Label(self.window, text="Draw something",
                               font=('Helvetica', 18, 'bold'), fg='#526CFF', bg=self.BACKGROUND_COLOR)
        self.lbl_title.grid(column=0, row=0, columnspan=6, pady=5)

        self.canvas = tk.Canvas(self.window, bg='white', width=560, height=560)
        self.canvas.grid(column=0, row=1, columnspan=6, padx=20)
        self.canvas.bind('<B1-Motion>', self.paint, add="+")
        self.canvas.bind('<B1-Motion>', self.fill_array, add="+")
        self.canvas.bind('<B1-Motion>', self.guess, add="+")
        self.canvas.bind('<ButtonRelease-1>', self.reset)

        self.btn_clear = ttk.Button(self.window, text="Clear",
                                    command=lambda: self.clear_canvas())
        self.btn_clear.grid(column=2, row=2, pady=10)

        self.btn_submit = ttk.Button(self.window, text="Submit",
                                command=lambda: self.submit_drawing())
        self.btn_submit.grid(column=3, row=2)

        self.lbl_prediction = tk.Label(self.window, text="I think you are drawing this :",
                                  font=('Helvetica', 18, 'bold'), fg='#526CFF', bg=self.BACKGROUND_COLOR)
        self.lbl_prediction.grid(column=7, row=1, columnspan=6, padx=50)

        self.old_x = None
        self.old_y = None
        self.line_width = 20
        self.line_color = "#000"
        self.array = 0 * np.ones(784, dtype=np.uint8)

        self.window.mainloop()

    def paint(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=self.line_color,
                               capstyle=tk.ROUND, smooth=tk.TRUE)
        self.old_x = event.x
        self.old_y = event.y

    def fill_array(self, event):
        pos = int(event.x / 20) + int(event.y / 20) * 28
        blur = 5
        if pos < 784 and self.canvas.winfo_width() - 1 > event.x > 0 \
                and self.canvas.winfo_height() - 1 > event.y > 0:
            self.array[pos] = 255
            if pos + 1 < 784:
                if self.array[pos + 1] + blur <= 255:
                    self.array[pos + 1] += blur
            if pos - 1 > 0:
                if self.array[pos - 1] + blur <= 255:
                    self.array[pos - 1] += blur
            if pos + 28 < 784:
                if self.array[pos + 28] + blur <= 255:
                    self.array[pos + 28] += blur
            if pos - 28 > 0:
                if self.array[pos - 28] + blur <= 255:
                    self.array[pos - 28] += blur

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def clear_canvas(self):
        self.canvas.delete('all')
        self.array = 0 * np.ones(784, dtype=np.uint8)

    def submit_drawing(self):
        imageHandling.array_to_img(self.array, "../output/drawing.png")
        # self.guess()

    def guess(self, event):
        end = time.time_ns()
        if (end - self.start) > 500000000: #0.5s
            self.start = time.time_ns()

            image = (np.expand_dims(self.array.reshape(28, 28), 0))
            # image = np.array([self.array.reshape(28, 28)])
            prediction = self.cnn.get_prediction(image)
            txt = "I think you are drawing this : \n" + self.cnn.class_names[np.argmax(prediction)]
            self.lbl_prediction.configure(text=txt)

            # plot.plot_prediction(prediction, self.cnn.class_names)




