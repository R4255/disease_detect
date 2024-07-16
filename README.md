Plant Disease Classification Model

Overview
This project leverages a deep learning approach to classify plant diseases using a Convolutional Neural Network (CNN) model trained on the New Plant Diseases Dataset from Kaggle. The model is capable of identifying 38 different classes of plant diseases, helping farmers and agricultural experts take timely action to protect crops.

Dataset
The dataset consists of thousands of labeled images of diseased and healthy plant leaves across various plant species. It is organized into training and validation sets, ensuring robust model performance.

Model Architecture
The CNN model was built using TensorFlow and Keras. It includes multiple convolutional layers, max-pooling layers, and dense layers to effectively learn and classify the complex features of plant diseases.

Setup and Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/plant-disease-classification.git
cd plant-disease-classification
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Download the dataset from Kaggle and place it in the appropriate directory.

Training the Model
To train the model, use the following command:

bash
Copy code
python train.py
Evaluating the Model
To evaluate the model's performance on the validation set, use:

bash
Copy code
python evaluate.py
Testing the Model
To test the model with a sample image:

bash
Copy code
python test.py --image_path path/to/image.jpg
Results
The model achieves high accuracy in classifying 38 different plant diseases, as demonstrated in the example below:

python
Copy code
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Load the trained model
cnn = tf.keras.models.load_model('trained_plant_disease_model.keras')

# Test Image Visualization
image_path = './test/AppleCedarRust1.JPG'
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.show()

# Preprocess and predict
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])
predictions = cnn.predict(input_arr)

result_index = np.argmax(predictions)
class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', ...] # truncated for brevity
model_prediction = class_name[result_index]

plt.imshow(img)
plt.title(f"Disease Name: {model_prediction}")
plt.xticks([])
plt.yticks([])
plt.show()
Contributing
Contributions are welcome! Please read the contributing guidelines before submitting a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Kaggle for providing the dataset.
TensorFlow and Keras for their powerful deep learning libraries.
