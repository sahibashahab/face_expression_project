import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import Tk
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam




emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

test_set = test_datagen.flow_from_directory(
    'test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

# emotion_cnn_model = Sequential()

# emotion_cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
# emotion_cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# emotion_cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_cnn_model.add(Dropout(0.25))

# emotion_cnn_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# emotion_cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_cnn_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
# emotion_cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
# emotion_cnn_model.add(Dropout(0.25))

# emotion_cnn_model.add(Flatten())
# emotion_cnn_model.add(Dense(1024, activation='relu'))
# emotion_cnn_model.add(Dropout(0.5))
# emotion_cnn_model.add(Dense(7, activation='softmax'))

# emotion_cnn_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])



# # cnn.fit(x=training_set,validation_data=test_set,epochs=30)
# # Train the neural network/model
# emotion_cnn_model.fit(
#         x=training_set,
#         steps_per_epoch=19638 // 64,
#         epochs=50,
#         validation_data=test_set,
#         validation_steps=7879 // 64)

# # save model structure in jason file
# model_json = emotion_cnn_model.to_json()
# with open("emotion_model.json", "w") as json_file:
#     json_file.write(model_json)

# # save trained model weight in .h5 file
# emotion_cnn_model.save('emotion_model.h5')

with open("emotion_model.json", "r") as json_file:
    loaded_model_json = json_file.read()

emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("emotion_model.h5")

root = Tk()
root.title("Emotion Detection")

frame = tk.Frame(root)
lbl_heading = tk.Label(frame, text='Emotion Detection', padx=25, pady=25, font=('verdana', 16))
lbl_pic_path = tk.Label(frame, text='Image Path:', padx=25, pady=25, font=('verdana', 16))
lbl_show_pic = tk.Label(frame)
entry_pic_path = tk.Entry(frame, font=('verdana', 14))
btn_browse = tk.Button(frame, text='Select Image', bg='grey', fg='#ffffff', font=('verdana', 14))
lbl_prediction = tk.Label(frame, text='Emotion: ', padx=25, pady=25, font=('verdana', 16))
lbl_predict = tk.Label(frame, font=('verdana', 16))
lbl_confid = tk.Label(frame, text='Confidence: ', padx=25, pady=25, font=('verdana', 16))
lbl_confidence = tk.Label(frame, font=('verdana', 16))

progress = Progressbar(frame, orient='horizontal', length=300, mode='determinate')
progress.grid(row=6, column=0, columnspan=2, pady=10)
progress_label = tk.Label(frame, font=('verdana', 16))
progress_label.grid(row=7, column=0, columnspan=2, pady=10)

def predict_emotion(img_path):
    img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = emotion_model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_class_index]
    confidence_score = prediction[0][predicted_class_index] * 100

    return predicted_emotion, confidence_score

def select_image():
    global img
    filename = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                          filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg *.jpeg")))
    if filename:
        img = Image.open(filename).convert("RGB")
        img = img.resize((250, 250), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)

        lbl_show_pic['image'] = img
        entry_pic_path.delete(0, tk.END)
        entry_pic_path.insert(0, filename)

        predicted_emotion, confidence = predict_emotion(filename)
        lbl_predict.config(text=predicted_emotion)
        lbl_confidence.config(text=f"{confidence:.2f}%")

        progress_value = int(confidence)
        progress['value'] = progress_value

        if progress_value >= 80:
            progress_color = 'green'
            confidence_label = 'High'
        else:
            progress_color = 'red'
            confidence_label = 'Low'

        progress_label.config(text=f'Confidence: {confidence_label}')

def use_webcam():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Live Emotion Detection")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        img_array = np.expand_dims(np.expand_dims(resized, axis=0), axis=-1)
        img_array = img_array / 255.0

        prediction = emotion_model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_emotion = emotion_labels[predicted_class_index]
        confidence_score = prediction[0][predicted_class_index] * 100

        label_text = f'{predicted_emotion} ({confidence_score:.2f}%)'
        color = (0, 255, 0) if confidence_score >= 80 else (0, 0, 255)

        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2, cv2.LINE_AA)

        cv2.imshow("Live Emotion Detection", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to close
            break

    cap.release()
    cv2.destroyAllWindows()

# Bind button commands
btn_browse.config(command=select_image)
btn_webcam = tk.Button(frame, text='Use Webcam', bg='blue', fg='#ffffff', font=('verdana', 14), command=use_webcam)

# Layout adjustments for left and right buttons
frame.pack()

lbl_heading.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
lbl_pic_path.grid(row=1, column=0)
entry_pic_path.grid(row=1, column=1, padx=(0, 20))
lbl_show_pic.grid(row=2, column=0, columnspan=2)
btn_browse.grid(row=3, column=0, padx=10, pady=10, sticky="w")  # Left side
btn_webcam.grid(row=3, column=1, padx=10, pady=10, sticky="e")  # Right side
lbl_prediction.grid(row=4, column=0)
lbl_predict.grid(row=4, column=1, padx=2, sticky='w')
lbl_confid.grid(row=5, column=0)
lbl_confidence.grid(row=5, column=1, padx=2, sticky='w')

root.mainloop()