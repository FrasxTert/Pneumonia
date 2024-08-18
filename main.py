import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

base_dir = '/content/drive/MyDrive/data science/pnevmoniya'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')


def create_dataframe(data_dir, is_test=False):
    categories = ['NORMAL', 'PNEUMONIA'] if not is_test else [None]
    file_paths = []
    labels = []

    for category in categories:
        if category:
            category_dir = os.path.join(data_dir, category)
            for filename in os.listdir(category_dir):
                if filename.endswith('.jpeg'):
                    file_paths.append(os.path.join(category_dir, filename))
                    labels.append(category)
        else:
            for filename in os.listdir(data_dir):
                if filename.endswith('.jpeg'):
                    file_paths.append(os.path.join(data_dir, filename))
                    labels.append(None)

    df = pd.DataFrame({
        'id': file_paths,
        'labels': labels
    })
    return df

def create_test_dataframe(data_dir):
    file_paths = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpeg'):
            file_paths.append(os.path.join(data_dir, filename))
    
    df = pd.DataFrame({
        'id': file_paths,
        'labels': [None] * len(file_paths)
    })
    return df


train_df = create_dataframe(train_dir)
test_df = create_test_dataframe(test_dir)

def encode_labels(df):
    df['labels'] = df['labels'].map({'NORMAL': 'normal', 'PNEUMONIA': 'pneumonia'})
    return df

train_df = encode_labels(train_df)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='id',
    y_col='labels',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='id',
    y_col=None,
    target_size=(150, 150),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=20,
)

predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size + 1)
predicted_labels = np.round(predictions).astype(int).flatten()
filenames = test_generator.filenames

results = pd.DataFrame({
    'id': filenames,
    'labels': predicted_labels
})

results.to_csv('/content/drive/MyDrive/data science/pneumonia.csv', index=False)
