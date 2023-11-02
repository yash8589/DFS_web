# Image Classification Model for Cancer Types

This README file provides an overview of an image classification model for classifying different types of cancer. The model uses the TensorFlow and Keras libraries to train a deep neural network and predict the class of input images. The model is based on the ResNet-50 architecture and is designed to classify images into five classes: brain menin, oral normal, oral squamous cell carcinoma (scc), brain glioma, and brain tumor. Below, we will explain the code step by step.

## Code Explanation:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import cv2
import imageio
import os
from tqdm import tqdm
import gc
import random
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
```

These are the necessary imports for the code, including libraries for data manipulation, visualization, image processing, and deep learning.

```python
brain_cancer_path = 'mcc2'
```

`brain_cancer_path` is the path to the directory containing the dataset.

```python
directories = []
for directory in os.listdir(brain_cancer_path):
    directories.append(directory)
print('Classes Present: ', list(directories))
```

This code lists the classes present in the dataset by iterating through the subdirectories in `brain_cancer_path`.

```python
all_pre_files = []
all_early_files = []
oral_scc_files = []
all_benign_files = []
oral_normal_files = []
all_pro_files = []
brain_glioma_files = []
brain_tumor_files = []

# Loop through directories and collect file paths for different classes.
```

The code collects file paths for each class within the dataset.

```python
print('Total all_pre_files: ', len(all_pre_files))
print('Total all_early_files: ', len(all_early_files))
print('Total oral_scc_files: ', len(oral_scc_files))
```

This code prints the total number of files for each class.

```python
random_num = random.randint(0, len(all_pre_files))
brain_tumor_pic = all_pre_files[random_num]
brain_early_pic = all_early_files[random_num]
oral_scc_pic = oral_scc_files[random_num]

brain_tumor_data = imageio.imread(brain_tumor_pic)
brain_early_data = imageio.imread(brain_early_pic)
oral_scc_data = imageio.imread(oral_scc_pic)

# Display example images from different classes.
```

This code randomly selects images from different classes and displays them for visualization.

```python
gc.collect()
```

This code performs garbage collection to free up memory.

```python
train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    'mcc2',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='both',
)
```

This code creates training and validation datasets from the image directory. It uses `tf.keras.utils.image_dataset_from_directory` to load and preprocess the images.

```python
checkpoint_filepath = 'checkpoint_small'
callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)
]
```

These are callbacks used during model training, including early stopping, learning rate reduction, and model checkpointing.

```python
base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model_resnet.layers:
    layer.trainable = False

x = base_model_resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model_resnet.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

This code defines the model architecture using the pre-trained ResNet-50 model as the base. It adds additional layers, including a Global Average Pooling layer and fully connected layers.

```python
history = model.fit(train_ds, verbose=1, epochs=1, batch_size=32, validation_data=val_ds, callbacks=callback)

model.save('mcc_latest.h5')
```

The model is trained on the training dataset with one epoch, and the best model is saved.

```python
predict = model.predict(val_ds)
predict
```

This code generates predictions on the validation dataset.

```python
from tensorflow.keras.models import load_model

# find the index of the maximum element in an array
def find_max(arr):
    maxi = max(arr)
    for i in range(len(arr)):
        if arr[i] == maxi:
            return i

# predict the class from an input image using our trained model
def predict_image_class(model, img, show=True):
    img = cv2.imread(img)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [1, 224, 224, 3])
    classes = model.predict(img)
    print(classes)

    max_ind = find_max(classes[0])
    print(train_ds.class_names[max_ind])
    if show:
        plt.imshow(img[0])
        plt.show()

model = load_model('mcc_latest.h5')
predict_image_class(model, 'mcc2/oral_normal/oral_normal_0035.jpg')
```

This code defines functions for predicting the class of an input image using the trained model and demonstrates how to use the model for prediction.

Make sure to replace `'mcc2'` with the actual path to your dataset directory and adapt the code according to your specific dataset structure and requirements. This README provides an overview of the code's functionality, and you may need to modify it to suit your needs.