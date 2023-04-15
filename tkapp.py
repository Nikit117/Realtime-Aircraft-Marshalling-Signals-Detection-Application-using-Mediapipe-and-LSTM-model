import cv2
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.figure import Figure
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tkinter import ttk
# from ttkthemes import ThemedStyle
import pyttsx3
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
print(tf.config.list_physical_devices('GPU'))
engine = pyttsx3.init()
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

DATA_PATH = os.path.join('MP_Data')

actions = np.array(["Despatch Aircraft","Hold Position","Normal Stop","OpenClose Stairs","Release Brakes","Set Brakes","Slow Down","Straight Ahead","Turn Left","Turn Right"])
no_sequences = 30
sequence_length = 90

colors = [(245,117,16), (117,245,16), (16,117,245),(117,16,245),(245,16,117),(16,200,78),(200,78,16),(78,16,200),(16,78,200),(78,200,16)]

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), int(90+num*40)), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

model = keras.Sequential()
model.add(keras.layers.LSTM(512, input_shape=(90, 1662), return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(256, return_sequences=True))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.LSTM(128))
model.add(keras.layers.BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.load_weights('action3.h5')

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = MyVideoCapture(self.video_source)
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        # Create the buttons using ttk
        self.btn_snapshot = ttk.Button(window, text='Snapshot', command=self.snapshot)
        self.play_pause_button = ttk.Button(window, text='Play', command=self.play_pause)
        self.change_video_source_button_to_directoy = ttk.Button(window, text='Change Video Source', command=self.change_video_source)
        self.sound_button_for_last_element_in_sentence = ttk.Button(window, text='Sound', command=self.sound)
        self.annot = ttk.Button(window, text='Annotate', command=self.annotate)
        # self.show_graph_button = ttk.Button(window, text='Show Graph', command=self.show_graph)

        # Add the buttons to the window
        self.change_video_source_button_to_directoy.pack(side='left', padx=20, pady=10)
        self.play_pause_button.pack(side='left', padx=20, pady=10)
        self.btn_snapshot.pack(side='left', padx=20, pady=10)
        self.sound_button_for_last_element_in_sentence.pack(side='left', padx=20, pady=10)
        self.annot.pack(side='left', padx=20, pady=10)
        # self.show_graph_button.pack(side='left', padx=20, pady=10)
        self.actions = ["Despatch Aircraft","Hold Position","Normal Stop","OpenClose Stairs","Release Brakes","Set Brakes","Slow Down","Straight Ahead","Turn Left","Turn Right"]
        
        # Start the update loop
        self.delay = 15
        self.update()
        self.window.mainloop()
        
    def snapshot(self):
        ret, frame = self.vid.get_frame()
        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
    def update(self):

        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        self.window.after(self.delay, self.update)

    def play_pause(self):
        if self.play_pause_button["text"] == "Play":
            self.play_pause_button["text"] = "Pause"
            self.delay = 0
        else:
            self.play_pause_button["text"] = "Play"
            self.delay = 15
    
    def change_video_source(self):
        if self.change_video_source_button_to_directoy["text"] == "Change Video Source":
            self.change_video_source_button_to_directoy["text"] = "Change Video Source to Webcam"
            filename = filedialog.askopenfilename(initialdir="/", title="Select a File", filetypes=(("Video files", "*.mp4 *.avi"), ("all files", "*.*")))
            self.vid = MyVideoCapture(filename)
            
        else:
            self.change_video_source_button_to_directoy["text"] = "Change Video Source"
            self.vid = MyVideoCapture(0)
    
    def sound(self):
        if self.sound_button_for_last_element_in_sentence["text"] == "Sound":
            self.sound_button_for_last_element_in_sentence["text"] = "Mute"
            self.vid.sound = False
        else:
            self.sound_button_for_last_element_in_sentence["text"] = "Sound"
            self.vid.sound = True
            engine.stop()
    
    def annotate(self):
        if self.annot["text"] == "Annotate":
            self.annot["text"] = "Hide"
            self.vid.annot = False
        else:
            self.annot["text"] = "Annotate"
            self.vid.annot = True
            
class MyVideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.sequence = []
        self.sentence = []
        self.threshold = 0.5
        self.sound = True
        self.annot = True
        self.prediction = None

    
    def get_frame(self):
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            if self.vid.isOpened():
                ret, frame = self.vid.read()
                if ret:
                    frame = cv2.resize(frame, (640, 480))
                    image, results = mediapipe_detection(frame, holistic)
                    if self.annot:
                        draw_styled_landmarks(image, results)
                    keypoints = extract_keypoints(results)
                    self.sequence.append(keypoints)
                    self.sequence = self.sequence[-sequence_length:]
                    if len(self.sequence) == sequence_length:
                        res = model.predict(np.expand_dims(self.sequence, axis=0))[0]
                        self.prediction = res
                        if res[np.argmax(res)] > self.threshold:
                            if len(self.sentence) > 0:
                                if actions[np.argmax(res)] != self.sentence[-1]:
                                    self.sentence.append(actions[np.argmax(res)])
                                    if self.sound:
                                        engine.say(actions[np.argmax(res)])
                                        engine.runAndWait()
                            else:
                                self.sentence.append(actions[np.argmax(res)])
                        if len(self.sentence) > 3:
                            self.sentence = self.sentence[-3:]
                        image = prob_viz(res, actions, image, colors)
                    cv2.rectangle(image, (0,0), (640,40), (245,117,16), -1)
                    cv2.putText(image, ' '.join(self.sentence), (3,30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                    
                    return (ret, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                self.video_source = 0
                self.vid = cv2.VideoCapture(self.video_source)
                return (ret, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return (ret, None)
    
            
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

App(tk.Tk(), "Tkinter and OpenCV")



