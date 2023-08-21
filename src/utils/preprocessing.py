from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def init_data_generator():
    data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    train_generator = data_generator.flow_from_directory(
        'dataset/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training')

    validation_generator = data_generator.flow_from_directory(
        'dataset/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation')

    data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    test_generator = data_generator.flow_from_directory(
        'dataset/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

    return train_generator, validation_generator, test_generator
