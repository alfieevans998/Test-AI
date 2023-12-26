import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

model = VGG16(weights='imagenet')

def classify_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)

    decoded_predictions = decode_predictions(predictions, top=3)[0]

    print(f"Классификация для изображения {file_path}:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions, 1):
        print(f"{i}. {label} ({score:.2f})")

image_file_path 
classify_image(image_file_path)
