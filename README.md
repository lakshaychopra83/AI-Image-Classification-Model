# AI-Image-Classification-Model
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Canvas
from PIL import Image, ImageTk  # Import the necessary module from Pillow

# Load the pre-trained MobileNetV2 model (you can use other models like VGG, ResNet, etc.)
model = MobileNetV2(weights='imagenet')

# Function to load, preprocess, and classify an image
def classify_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
    if file_path:
        img = Image.open(file_path)  # Use Pillow to open the image
        img = img.resize((224, 224))  # Resize the image to MobileNetV2's input size
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        # Make predictions
        predictions = model.predict(img)

        # Decode and display the top 5 predicted classes
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        prediction_text = "\n".join([f"{label}: {score:.2f}" for (_, label, score) in decoded_predictions])
        result_label.config(text=prediction_text)

        # Display the selected image
        img = Image.open(file_path)
        img = img.resize((300, 300))  # Resize the image for display
        img = ImageTk.PhotoImage(img)
        image_canvas.img = img  # Keep a reference to prevent image from being garbage collected
        image_canvas.create_image(0, 0, anchor=tk.NW, image=img)

# Main function
def main():
    root = tk.Tk()
    root.title("Image Classification App")

    # Create a button to open the file dialog and classify the image
    classify_button = tk.Button(root, text="Upload Image", command=classify_image)
    classify_button.pack()

    # Create a label to display the predictions
    global result_label
    result_label = Label(root, text="", wraplength=300, justify="left")
    result_label.pack()

    # Create a canvas to display the selected image
    global image_canvas
    image_canvas = Canvas(root, width=300, height=300)
    image_canvas.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
