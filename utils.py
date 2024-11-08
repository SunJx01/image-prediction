# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.models import load_model
# from keras import layers
# from keras.models import Sequential
# import keras
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from keras.preprocessing.image import img_to_array
# import io
# from keras.applications import MobileNet
# from keras.applications.mobilenet import preprocess_input
# from keras.preprocessing.image import load_img, img_to_array

# class_names = [
#     "airplane",
#     "automobile",
#     "bird",
#     "cat",
#     "deer",
#     "dog",
#     "frog",
#     "horse",
#     "ship",
#     "truck",
# ]


# def get_dataset_info():
#     # loading and spliting dataset
#     (X_train, y_train), (X_test, y_test) = cifar10.load_data()
#     return {
#         # size & quantity of data
#         "X_train: " + str(X_train.shape),
#         "Y_train: " + str(y_train.shape),
#         "X_test:  " + str(X_test.shape),
#         "Y_test:  " + str(y_test.shape),
#     }


# def load_preprocess_image_cnn(img_pil):
#     # Use BytesIO to treat the byte data as a file-like object
#     img = img_pil
#     # img = load_img(img, target_size=(32, 32))  # Adjust target_size to match the input size used during training

#     # Resize the image to the size required by the model (e.g., 32x32)
#     img = img.resize((32, 32))

#     # Convert the image to an array and normalize it
#     img_array = img_to_array(img) / 255.0

#     # CNN expects a 3D array input per image (height, width, channels)
#     img_array = img_array.reshape(1, 32, 32, 3)  # Reshape to include batch dimension

#     return img_array


# def load_preprocess_image_tl(img_pil):
#     # Load the image
#     img = img_pil

#     img = img.resize((32, 32))

#     # Convert the image to an array
#     img_array = img_to_array(img)
#     # Apply MobileNet specific preprocessing
#     img_array = preprocess_input(
#         img_array
#     )  # Preprocess the image data to the format the model expects
#     # Reshape for the model (adding batch dimension)
#     img_array = img_array.reshape(1, 32, 32, 3)
#     return img_array


# # def predict(preprocessed_image_cnn):
# #     # Ensure you are using the correct CNN model here

# #     model = get_model()
# #     prediction_cnn = model.predict(
# #         preprocessed_image_cnn
# #     )  # Assuming 'model' is your CNN
# #     predicted_class_cnn = np.argmax(prediction_cnn, axis=1)
# #     predicted_class_name_cnn = class_names[predicted_class_cnn[0]]
# #     return predicted_class_name_cnn


# def predict_tl(preprocessed_image_tl):
   
#     model_loaded = load_model('trained_model.h5')
    
#     # model_tl = get_model()
#     prediction = model_loaded.predict(preprocessed_image_tl)

#     predicted_class_tl = np.argmax(prediction, axis=1)
#     predicted_class_name_tl = class_names[predicted_class_tl[0]]
#     # prediction_tl = model_tl.predict(preprocessed_image_tl)
#     # predicted_class_tl = np.argmax(prediction_tl, axis=1)
#     # predicted_class_name_tl = class_names[predicted_class_tl[0]]

#     return predicted_class_name_tl


# def get_result_image(img_pil, predicted_class_name_cnn):
#     # Display the image and the prediction
#     img = img_pil
#     # Display the image and the prediction
#     plt.imshow(img)
#     plt.title(f"Predicted by CNN: {predicted_class_name_cnn}")
#     plt.axis("off")
#     print(f"Predicted class by CNN: {predicted_class_name_cnn}")

#     # Save to BytesIO object
#     buf = io.BytesIO()
#     plt.savefig(buf, format="png")
#     plt.close()  # Close the plot to avoid memory issues
#     buf.seek(0)  # Rewind the buffer to start
#     return buf


# def get_result_image_tl(img_pil, predicted_class_name_tl):
#     img = img_pil

#     # Display the image and the prediction
#     plt.imshow(img)
#     plt.title(f"Predicted by Transfer Learning: {predicted_class_name_tl}")
#     plt.axis("off")
#     # plt.show()

#     print(f"Predicted class by Transfer Learning: {predicted_class_name_tl}")

#     # Save to BytesIO object
#     buf = io.BytesIO()
#     plt.savefig(buf, format="png")
#     plt.close()  # Close the plot to avoid memory issues
#     buf.seek(0)  # Rewind the buffer to start
#     return buf


# def get_model():

#     (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
#     base_model = MobileNet(
#         weights="imagenet", include_top=False, input_shape=(32, 32, 3)
#     )
#     x = base_model.output
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dense(512, activation="relu")(x)
#     predictions = layers.Dense(10, activation="softmax")(x)
#     model_tl = keras.models.Model(inputs=base_model.input, outputs=predictions)

#     model_tl.compile(
#         optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
#     )
#     ## Model Building:

#     history_tl = model_tl.fit(
#         X_train, y_train, validation_data=(X_test, y_test), epochs=10
#     )
#     model_tl.save('trained_model.h5')

#     return model_tl




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

# Define the class names for the CIFAR-10 dataset
class_names = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
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

def load_preprocess_image(img_pil):
    """
    Preprocess the input image for prediction.
    Resizes the image to 32x32, converts it to an array, normalizes,
    and reshapes it to fit the model input dimensions.
    """
    img = img_pil.resize((32, 32))
    img_array = img_to_array(img) / 255.0  # Normalize the pixel values
    img_array = img_array.reshape(1, 32, 32, 3)  # Add batch dimension
    return img_array

def predict_image(img_pil):
    """
    Load the pre-trained model, preprocess the input image,
    and make a prediction on the image class.
    """
    # Load pre-trained model
    model = load_model('trained_model.h5')
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
    base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    
    # Add custom layers on top
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    predictions = layers.Dense(10, activation="softmax")(x)
    model = keras.models.Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

    # Save the trained model
    model.save('trained_model.h5')
    return model
