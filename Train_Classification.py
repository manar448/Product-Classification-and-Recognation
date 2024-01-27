import pickle

import cv2
import os

import joblib
import numpy as np
from sklearn.decomposition import PCA
from keras_preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import time
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

start_time_train = time.time()
data_path = "C:/Users/Malak/Documents/Semester 7/Computer Vision/Project/Data/Product Classification"


# Preprocessing
def read(data_path, file_name):
    image_paths = []
    labels = []
    for class_label in range(1, 21):
        class_folder = str(class_label)
        class_path = os.path.join(data_path, class_folder, file_name)
        label = class_label
        # Iterate through images in the current class folder
        for filename in os.listdir(class_path):
            image_path = os.path.join(class_path, filename)
            image_paths.append(image_path)
            labels.append(label)
    return image_paths, labels


def preprocess_image(image_path, target_size=(224, 224), use_pca=False):
    # Read the image
    original_image = cv2.imread(image_path)
    # Resize the image
    resized_image = cv2.resize(original_image, target_size)
    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to [0, 1]
    normalized_image = rgb_image.astype('float32') / 255.0

    # Flatten the image if using PCA
    if use_pca:
        # flatten the image to a 1D array
        flattened_image = normalized_image.reshape(-1, 3)

        # Standardize pixel values
        scaler = StandardScaler()
        standardized_image = scaler.fit_transform(flattened_image)

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=3)
        reduced_image = pca.fit_transform(standardized_image)

        # Reshape back to the original shape
        reduced_image = reduced_image.reshape(target_size[0], target_size[1], 3)
        return reduced_image
    else:
        return normalized_image


# Extract HOG features
def extract_hog_features(images):
    features = []
    for image in images:
        # Assuming images are grayscale
        hog_feature = hog(image, orientations=8, pixels_per_cell=(16, 16),
                          cells_per_block=(1, 1), visualize=True)[0]
        features.append(hog_feature.ravel())

    return features


def data_preparation(image_paths, labels, augment=False, type_model='classical'):
    p_images = []
    if type_model == 'CNN' or type_model == 'cnn':
        for img in image_paths:
            preprocessed_image = preprocess_image(img, use_pca=False)
            p_images.append(preprocessed_image)

        if augment:
            # Data augmentation: create an instance of ImageDataGenerator
            datagen = ImageDataGenerator(
                rotation_range=20,  # Randomly rotate images by up to 20 degrees
                width_shift_range=0.2,  # Randomly shift images horizontally by up to 20%
                height_shift_range=0.2,  # Randomly shift images vertically by up to 20%
                shear_range=0.2,  # Apply shear transformations
                zoom_range=0.2,  # Randomly zoom in on images by up to 20%
                horizontal_flip=True,  # Randomly flip images horizontally
                vertical_flip=True,  # Randomly flip images vertically
                fill_mode='nearest'  # Strategy for filling in newly created pixels after rotation or shifting
            )

            # Convert images to numpy array
            p_images_array = np.array(p_images)

            # Apply data augmentation to the images
            augmented_images = []
            for batch in datagen.flow(p_images_array, batch_size=len(p_images_array), shuffle=False):
                augmented_images.extend(batch)
                break  # Only apply augmentation once

            # Convert augmented images back to list
            augmented_images = list(augmented_images)
            # print("images after augmentation: ", len(augmented_images))
            # Append augmented features and labels
            # hog_features += extract_hog_features(augmented_images)
            p_images += augmented_images
            augmented_labels = labels.copy()
            augmented_labels += labels
            # print("hog after: ", len(hog_features))
            return np.array(p_images), np.array(augmented_labels)

        return np.array(p_images), np.array(labels)
    else:
        for img in image_paths:
            preprocessed_image = preprocess_image(img, use_pca=True)
            p_images.append(preprocessed_image)

        # print("images before augmentation: ", len(p_images))
        # Extract HOG features
        hog_features = extract_hog_features(p_images)
        # print("hog features: ", len(hog_features))
        if augment:
            # Data augmentation: create an instance of ImageDataGenerator
            datagen = ImageDataGenerator(
                rotation_range=20,  # Randomly rotate images by up to 20 degrees
                width_shift_range=0.2,  # Randomly shift images horizontally by up to 20%
                height_shift_range=0.2,  # Randomly shift images vertically by up to 20%
                shear_range=0.2,  # Apply shear transformations
                zoom_range=0.2,  # Randomly zoom in on images by up to 20%
                horizontal_flip=True,  # Randomly flip images horizontally
                vertical_flip=True,  # Randomly flip images vertically
                fill_mode='nearest'  # Strategy for filling in newly created pixels after rotation or shifting
            )

            # Convert images to numpy array
            p_images_array = np.array(p_images)

            # Apply data augmentation to the images
            augmented_images = []
            for batch in datagen.flow(p_images_array, batch_size=len(p_images_array), shuffle=False):
                augmented_images.extend(batch)
                break  # Only apply augmentation once

            # Convert augmented images back to list
            augmented_images = list(augmented_images)
            # print("images after augmentation: ", len(augmented_images))
            # Append augmented features and labels
            hog_features += extract_hog_features(augmented_images)
            augmented_labels = labels.copy()
            augmented_labels += labels
            # print("hog after: ", len(hog_features))
            return np.array(hog_features), np.array(augmented_labels)

        return np.array(hog_features), np.array(labels)


def SVM(X_train, Y_train):
    start = time.time()
    # kernel ==> (linear, poly, rbf, sigmoid)
    # (1.0, linear, scale)==> 26.47%, (0.1, linear, scale)==> 26.47%
    # (10, linear, 1)==> 26.47%, (10, poly, 1) ==> 29.41%
    svm = SVC(C=10, kernel='linear', gamma=1, random_state=42)

    svm.fit(X_train, Y_train)

    # Save the  SVM model to a file
    with open('SVM model.pkl', 'wb') as f:
        pickle.dump(svm, f)

    end = time.time()
    print(f"Time Train of SVM: {((end - start) / 60) :.2f}")


def logistic_regression(X_train, Y_train):
    start = time.time()
    # Define the parameter grid
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
                  'penalty': ['l1', 'l2']}

    # Create a logistic regression model
    logreg = LogisticRegression(random_state=0)

    # Create a GridSearchCV object
    grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')

    # Fit the grid search to the data
    grid_search.fit(X_train, Y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    # print("Best Parameters:", best_params)

    # Use the best parameters to train your final model
    best_logreg = LogisticRegression(random_state=0, **best_params)
    best_logreg.fit(X_train, Y_train)

    # Save the  logistic regression model to a file
    with open('logistic model.pkl', 'wb') as f:
        pickle.dump(best_logreg, f)

    print(f"Best Hyperparameters of Logistic Regression: {best_params}")
    end = time.time()
    print(f"Time Train of Logistic Regression: {((end - start) / 60) :.2f}")


def KNN(X_train, Y_train):
    start = time.time()
    # Define the parameter grid
    param_grid = {
        'n_neighbors': [1, 3, 5, 7],
        'metric': ['euclidean', 'manhattan'],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    # Create a KNN classifier
    knn = KNeighborsClassifier()

    # Create a GridSearchCV object
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

    # Fit the grid search to the data
    grid_search.fit(X_train, Y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    # print("Best Parameters:", best_params)
    # Use the best parameters to train your final model
    best_knn = KNeighborsClassifier(**best_params)
    best_knn.fit(X_train, Y_train)

    # Save the  KNN model to a file
    with open('KNN model.pkl', 'wb') as f:
        pickle.dump(best_knn, f)

    print(f"Best Hyperparameters of KNN: {best_params}")
    end = time.time()
    print(f"Time Train of KNN: {((end - start) / 60) :.2f}")


def random_forest(x_train, y_train):
    start = time.time()
    # Initialize and train the Random Forest classifier
    rand = RandomForestClassifier(n_estimators=100, random_state=42)
    rand.fit(x_train, y_train)

    # Save the  random model to a file
    with open('random model.pkl', 'wb') as f:
        pickle.dump(rand, f)

    end = time.time()
    print(f"Time Train of Random Forest: {((end - start) / 60) :.2f}")


def training_classical(image_paths, labels, type_model):
    # data preparation for classical models (Apply data augmentation only to the training set)
    hog_features, augmented_labels = data_preparation(image_paths, labels, augment=False, type_model=type_model)
    # hog_features_val, val_labels = data_preparation(image_paths_val, labels_val, augment=False, type_model=type_model)

    # Flatten the HOG features
    hog_features_flat = hog_features.reshape(hog_features.shape[0], -1)
    # hog_features_val_flat = hog_features_val.reshape(hog_features_val.shape[0], -1)

    '''with open('HOG_Features_train.pkl', 'wb') as f:
        pickle.dump(hog_features_flat, f)'''

    SVM(hog_features_flat, augmented_labels)
    logistic_regression(hog_features_flat, augmented_labels)
    KNN(hog_features_flat, augmented_labels)
    random_forest(hog_features_flat, augmented_labels)


def train_cnn_model(x_train, y_train, height, width, num_channels, num_classes, num_epochs, model_save_path):
    start = time.time()
    np.random.seed(42)
    tf.random.set_seed(42)
    torch.manual_seed(42)

    model = Sequential()
    model.add(Conv2D(64, 3, padding="same", activation="relu", input_shape=(height, width, num_channels)))
    model.add(MaxPool2D())

    model.add(Conv2D(128, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())

    model.add(Conv2D(256, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())

    model.add(Conv2D(512, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    print("\n--------------------------------------------------------------CNN Model----------------"
          "-----------------------------------------------------------------")
    model.summary()
    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(x_train)

    # Split data into training and validation sets
    validation_split = 0.2
    num_samples = len(x_train)
    split_index = int(validation_split * num_samples)

    x_train_split = x_train[:-split_index]
    y_train_split = y_train[:-split_index]
    x_val_split = x_train[-split_index:]
    y_val_split = y_train[-split_index:]

    # Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Learning Rate Scheduler
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

    history = model.fit(datagen.flow(x_train_split, y_train_split, batch_size=32),
                        epochs=num_epochs,
                        validation_data=(x_val_split, y_val_split),
                        callbacks=[early_stopping, reduce_lr])

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Save the model weights
    model.save_weights(model_save_path + '_weights.h5')

    # Save the training history
    with open(model_save_path + '_history.pkl', 'wb') as history_file:
        history_data = {'accuracy': acc, 'val_accuracy': val_acc, 'loss': loss, 'val_loss': val_loss}
        joblib.dump(history_data, history_file)

    end = time.time()
    print(f"Time Train of CNN: {((end - start) / 60) :.2f}")


#############################################################################


# read data
image_paths, labels = read(data_path, 'Train')

# Classical Models (SVM ,LogisticRegression, KNN, RandomForest)
training_classical(image_paths, labels, 'classical')

# data preparation for CNN Model
images_train, train_labels = data_preparation(image_paths, labels, augment=True, type_model='CNN')

# CNN Model (using augmentation)
model_save_path = './saved_model/model.h5'
train_cnn_model(images_train, train_labels, 224, 224, 3, 21, 20, model_save_path)


end_time_train = time.time()
time_train = (end_time_train - start_time_train) / 60
print(f"Time of Train: {time_train :.2f}")

print("\n---------------------------------------------------------------------Train Done----------------------------------------------------------\n")
# Reset the warning filter to its default behavior
warnings.resetwarnings()
