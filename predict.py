import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

model = tf.keras.models.load_model('Flowers_Recognition(VGG16).h5')
class_labels = ['aster', 'daisy', 'dandelion', 'iris', 'jasmine',
                'lavender', 'marigold', 'rose', 'sunflower', 'tulip']

selected_image_path = None

def predict_image():
    global selected_image_path
    if not selected_image_path:
        messagebox.showwarning("No Image", "Please upload an image first.")
        return

    result_label.config(text="⏳ Predicting...", fg="black")
    root.update_idletasks()

    img = load_img(selected_image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    processed_img = np.expand_dims(img_array, axis=0)

    predictions = model.predict(processed_img)[0]
    class_index = np.argmax(predictions)
    confidence = predictions[class_index] * 100
    predicted_class = class_labels[class_index]

    result_label.config(
        text=f"🌼 {predicted_class.upper()} ({confidence:.2f}%)",
        fg="#2a9d8f"
    )

def load_image():
    global selected_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    selected_image_path = file_path

    img = Image.open(file_path)
    img = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    image_panel.config(image=img_tk)
    image_panel.image = img_tk
    result_label.config(text="", fg="black")

def reset_app():
    global selected_image_path
    selected_image_path = None
    image_panel.config(image=default_img)
    image_panel.image = default_img
    result_label.config(text="", fg="black")


root = tk.Tk()
root.title("🌸 Flower Classifier")
root.geometry("520x680")
root.resizable(False, False)


bg_image = Image.open("background_flower.jpg") if os.path.exists("background_flowera.jpg") else None
if bg_image:
    bg_image = bg_image.resize((520, 680))
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)


title = tk.Label(root, text="🌸 Flower Classifier", font=("Helvetica", 22, "bold"), bg="#ffffff", fg="#264653")
title.pack(pady=20)

frame = tk.Frame(root, bg="#ffffff", bd=2, relief="ridge")
frame.pack(pady=10)


placeholder = Image.new('RGB', (250, 250), color='lightgray')
default_img = ImageTk.PhotoImage(placeholder)
image_panel = tk.Label(frame, image=default_img, width=250, height=250)
image_panel.pack()

def styled_button(master, text, command, bg):
    return tk.Button(
        master, text=text, command=command,
        font=("Arial", 12, "bold"), bg=bg,
        fg="white", activebackground="#1d3557",
        activeforeground="white", bd=0,
        padx=20, pady=10, cursor="hand2"
    )

upload_btn = styled_button(root, "📁 Upload Image", load_image, "#118ab2")
upload_btn.pack(pady=10)

predict_btn = styled_button(root, "🔍 Predict", predict_image, "#06d6a0")
predict_btn.pack(pady=5)

reset_btn = styled_button(root, "🔄 Reset", reset_app, "#ef476f")
reset_btn.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 16, "bold"), bg="#ffffff")
result_label.pack(pady=20)

root.mainloop()       