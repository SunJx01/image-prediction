from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from keras import layers
from keras.models import Sequential
import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
import io
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from PIL import Image

a = Image.Image
# Define the class names for the CIFAR-10 dataset
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_dataset_info():
    """
    Load and return CIFAR-10 dataset information including the size and shape
    of the training and test data.
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    return {
        "X_train": str(X_train.shape),
        "Y_train": str(y_train.shape),
        "X_test": str(X_test.shape),
        "Y_test": str(y_test.shape),
    }


def load_preprocess_image(img_pil: Image.Image):
    # Ensure the image is in RGB format

    print(type(img_pil), "typeee2")

    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")

    # Resize the image to the expected input size for the model
    img = img_pil.resize((32, 32))

    # Convert the image to an array and check shape
    img_array = img_to_array(img)
    print(f"Image array shape after conversion: {img_array.shape}")  # Debug statement

    # Apply MobileNet specific preprocessing
    img_array = preprocess_input(img_array)  # Adjusts pixel values to expected range

    # Add a batch dimension for model prediction
    img_array = img_array.reshape(1, 32, 32, 3)
    return img_array


def predict_image(img_pil):
    """
    Load the pre-trained model, preprocess the input image,
    and make a prediction on the image class.
    """
    # Load pre-trained model
    model = load_model("trained_model.h5")
    # Preprocess image for prediction
    preprocessed_image = load_preprocess_image(img_pil)
    # Predict the class
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_class_name = class_names[predicted_class[0]]
    return predicted_class_name


def display_prediction(img_pil, predicted_class_name):
    """
    Display the image along with its predicted class label.
    Saves the plot in a buffer to allow further usage.
    """
    plt.imshow(img_pil)
    plt.title(f"Predicted Class: {predicted_class_name}")
    plt.axis("off")
    print(f"Predicted Class: {predicted_class_name}")

    # Save plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)  # Reset buffer position
    return buf


def build_and_save_model():
    """
    Build, compile, and train a MobileNet model on the CIFAR-10 dataset
    with a custom dense layer on top, then save the trained model to disk.
    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Load MobileNet with pre-trained ImageNet weights, excluding the top layer
    base_model = MobileNet(
        weights="imagenet", include_top=False, input_shape=(32, 32, 3)
    )

    # Add custom layers on top
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    predictions = layers.Dense(10, activation="softmax")(x)
    model = keras.models.Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

    # Save the trained model
    model.save("trained_model.h5")
    return model
