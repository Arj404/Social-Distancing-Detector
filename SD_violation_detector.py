import matplotlib.style as style
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Detection import social_distancing_config as config
from Detection.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
import time
from tkinter import *
from PIL import ImageTk, Image
import threading
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


def detect(frame):
    frame = imutils.resize(frame, width=400)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

    violate = set()

    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        height = np.array([r[1][3]-r[1][1] for r in results])
        min_distance = np.array([(h)/3 for h in height])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                distance = min_distance[i]+min_distance[j]
                if D[i, j] < distance:
                    violate.add(i)
                    violate.add(j)

    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        if i in violate:
            color = (0, 0, 255)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)
        cv2.circle(frame, (cX, cY), 3, color, 1)

    return frame, len(violate)


def videoLoop(panel1, panel2, vs):
    if vs:
        try:
            while not stopEvent.is_set():
                count = 0
                while vs.isOpened():
                    (grabbed, frame) = vs.read()
                    count = count + 1
                    if count % 10 != 0:
                        continue

                    if not grabbed:
                        break

                    frame, violation = detect(frame)
                    print(violation)
                    violationNum.append(violation)

                    frame = imutils.resize(frame, width=700)
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    image = ImageTk.PhotoImage(image)

                    if panel1 is None:
                        panel1 = Label(image=image)
                        panel1.image = image
                        panel1.grid(row=1, column=0, padx=10,
                                    pady=10, rowspan=3, columnspan=2)
                        # ,side="left", padx=10, pady=10
                        panel2.configure(text=f'Violations = {violation}')
                        panel2.text = f'Violations = {violation}'

                    else:
                        panel1.configure(image=image)
                        panel1.image = image
                        panel2.configure(text=f'Violations = {violation}')
                        panel2.text = f'Violations = {violation}'

                if vs.read()[0] == 0:
                    np.save('ViolationCount1.npy', violationNum)
                    print("Violation Count saved")
                    panel1.destroy()
                    panel1 = Label(text="Video Finished")
                    panel1.text = "Video Finished"
                    panel1.configure(font=("Avenir", 20),
                                     bg='#141414', fg="#FCA311")
                    panel1.grid(row=0, column=0, padx=20,
                                pady=20, rowspan=3, columnspan=2)
                    stopEvent.set()
                    print("resetting")
                    reset()
        except RuntimeError as e:
            print(f"[INFO] caught a RuntimeError: {e}")


def Analysis():
    np.save('ViolationCount1.npy', violationNum)
    print("Violation Count saved")
    if stopEvent:
        stopEvent.set()
    root.protocol("WM_DELETE_WINDOW", lambda arg=vs: onClose1(arg))
    violationCount = np.load('./ViolationCount.npy')
    # print(violationCount)

    style.use('seaborn-whitegrid')
    # style.use('ggplot')

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True,
                                   dpi=100, gridspec_kw={'height_ratios': [1, 1]})
    fig.set_size_inches(7, 4, forward=True)
    ax2.axis('equal')

    ax2 = plt.subplot(212)
    ax2.boxplot(violationCount,vert=False, )
    ax2.set(xlim=(0, 25))
    
    ax1 = plt.subplot(211)
    ax1.plot(np.arange(len(violationCount)), violationCount,
             color='red')
    ax1.set(xlabel='time', ylabel='no of violations')

    panel1 = FigureCanvasTkAgg(fig, root)
    panel1.get_tk_widget().grid(row=1, column=0, padx=10,
                                pady=10, rowspan=3, columnspan=2)
    panel1.draw()


def onClose1(vs):
    root.quit()


def onClose2(vs):
    stopEvent.set()
    vs.release()
    root.quit()


def onclick1():
    path = textfield.get()
    print(path)
    if path:
        if path == '0':
            path = 0
        vs = cv2.VideoCapture(path)
        if vs is None or not vs.isOpened():
            print('Warning: unable to open video source: ', path)
        else:
            root.protocol("WM_DELETE_WINDOW", lambda arg=vs: onClose2(arg))
            global thread
            global stopEvent
            stopEvent = threading.Event()
            thread = threading.Thread(
                target=videoLoop, args=(panel1, panel2, vs))
            thread.setDaemon(True)
            thread.start()


def onclick2():
    Analysis()


def reset():
    root.protocol("WM_DELETE_WINDOW", lambda arg=vs: onClose1(arg))
    print("bleh")
    global Label0
    global textfield
    global button1
    Label0.destroy()
    textfield.destroy()
    button1.destroy()

    Label0 = Label(root, text='Enter the File path')
    Label0.configure(font=("Avenir", 15), bg='#141414', fg="#FCA311")
    Label0.grid(row=2, column=0, padx=50)
    textfield = Entry(root, width=50, bg='#FCA311', fg='#141414')
    textfield.grid(row=2, column=1, padx=(10, 50), pady=40,)
    textfield.insert(0, "pedestrians.mp4")
    button1 = Button(root, text='START', padx=30, pady=5,
                     command=onclick1)
    button1.configure(font=("Avenir", 15), fg="#141414",
                      highlightbackground='#FCA311', relief=GROOVE, borderwidth=.1)
    button1.grid(row=3, column=0, columnspan=2)


start = time.time()

# initialing Model
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# initialising Video
violationNum = []
path = None
vs = None


# GUI Window Code
thread = None
stopEvent = None
root = Tk()
root.configure(bg='#141414')
root.wm_title("Social Distancing Detector")
root.geometry("1000x500")
root.resizable(width=True, height=True)
frame = None
panel1 = None
Label0 = None

# Initializing Video Gui
Label0 = Label(root, text='Enter the File path')
Label0.configure(font=("Avenir", 15), bg='#141414', fg="#FCA311")
Label0.grid(row=1, column=0, padx=50)
textfield = Entry(root, width=40, bg='#FCA311', fg='#141414')
textfield.grid(row=1, column=1, padx=(10, 50), pady=30,)
textfield.insert(0, "pedestrians.mp4")
button1 = Button(root, text='START', padx=30, pady=5,
                 command=onclick1)
button1.configure(font=("Avenir", 15), fg="#141414",
                  highlightbackground='#FCA311', relief=GROOVE, borderwidth=.1)
button1.grid(row=2, column=0, columnspan=2)

# Other Gui
panel2 = Label(text=f'Violations = None')
panel2.text = f'Violations = None'
panel2.configure(font=("Avenir", 20), bg='#141414', fg="#FCA311")
panel2.grid(row=1, column=2, padx=10, pady=10)

Label1 = Label(root, text='Social Distancing Detector')
Label1.configure(font=("Avenir", 20), bg='#141414', fg="#FCA311")
Label1.grid(row=0, columnspan=3)

button2 = Button(root, text='Analysis', padx=30, pady=5,
                 command=onclick2)
button2.configure(font=("Avenir", 15), fg="#141414",
                  highlightbackground='#FCA311', relief=GROOVE, borderwidth=.1)
button2.grid(row=3, column=2)


root.protocol("WM_DELETE_WINDOW", lambda arg=vs: onClose1(arg))


root.mainloop()
print(f"time taken = {time.time()-start}")
# pedestrians.mp4
