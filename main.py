import cv2
from tkinter import *
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import *
import tensorflow_datasets as tfds
from PIL import Image, ImageTk
import sklearn
from sklearn.model_selection import train_test_split
import os
import random

window = Tk()
window.resizable(False, False)
window.title("Rock Paper Scissors via Camera")
stream = cv2.VideoCapture(0)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
	'Data\\train',
	validation_split=0.2,
	subset="training",
	seed=123,
	image_size=(300, 300))

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
	'Data\\train',
	validation_split=0.2,
	subset="validation",
	seed=123,
	image_size=(300, 300)),
	

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
	'Data\\test',
	validation_split=0.01,
	subset="training",
	seed=123,
	image_size=(300, 300))

def create_model(a, b):
	m = Sequential([
		layers.experimental.preprocessing.Resizing(128, 128),
		layers.experimental.preprocessing.Rescaling(1./255, input_shape=(128, 128, 3)),
		layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
		layers.experimental.preprocessing.RandomRotation(0.2),
		layers.Conv2D(32, 3, padding='same', activation='relu'),
		layers.AveragePooling2D(),
		layers.Conv2D(16, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Conv2D(16, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Conv2D(8, 3, padding='same', activation='tanh'),
		layers.AveragePooling2D(),
		layers.Flatten(),
		layers.Dense(128, activation='relu'),
		layers.Dense(64, activation='relu'),
		layers.Dense(3, activation="softmax")
	])
	m.compile(
		optimizer='adam',
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)

	history = m.fit(
		a,
		validation_data=b,
		epochs=3
	)
	
	return m

def show_frame():
	_, frame = stream.read()
	frame = cv2.flip(frame, 1)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = Image.fromarray(frame)
	frame = ImageTk.PhotoImage(image=frame)
	lmain.imagetk = frame
	lmain.configure(image=frame)
	lmain.after(10, show_frame)


def take_image():
	_, frame = stream.read()
	frame = cv2.flip(frame, 1)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	out = cv2.imwrite('Capture\\capture.png', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	print(model.evaluate(test_ds))
	guess = model.predict(np.array([frame]))
	g = max(guess[0])
	guess = "Paper" if g == guess[0][0] else ("Rock" if g == guess[0][1] else "Scissors")
	randnum = random.randint(1, 3)
	comp = "Paper" if randnum == 1 else ("Rock" if randnum == 2 else "Scissors") 
	result = ""
	if comp == guess: 
		result = "Tie"
	elif (guess == "Paper" and comp == "Scissors") or (guess == "Rock" and comp == "Paper") or (guess == "Scissors" and comp == "Rock"):
		result = "You Lose"
	else:
		result = "You Win"

	lbot.configure(text=f"Your play was: {guess}. The computer's play was {comp}. {result}")


model = create_model(train_ds, val_ds)

lmain = Label()
lmain.pack()

lbot = Label()
lbot.pack()

snapshot = Button(text="Play", command=take_image)
snapshot.pack()

show_frame()
window.lift()
window.mainloop()
