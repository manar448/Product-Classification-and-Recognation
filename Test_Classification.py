from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns
from Train_Classification import *
import time
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

start_time_test = time.time()
# data_path = "C:/Users/Malak/Documents/Semester 7/Computer Vision/Project/Data/Product Classification"
data_path = "C:/Users/Malak/Desktop/SC Data/SC Data/Test Samples Classification"


def SVM(X_test, Y_test):
    start = time.time()
    with open('SVM model.pkl', 'rb') as f:
        svm = pickle.load(f)
    y_pred = svm.predict(X_test)

    accuracy = accuracy_score(Y_test, y_pred)
    print(f"Test Accuracy of SVM: {accuracy * 100:.2f}%")

    # visualize_classical_model(svm, X_test, Y_test, 'SVM')

    end = time.time()
    print(f"Time Test of SVM: {((end - start) / 60) :.2f}")


def logistic_regression(X_test, Y_test):
    start = time.time()
    with open('logistic model.pkl', 'rb') as f:
        best_logreg = pickle.load(f)

    # Make predictions on the test set
    y_pred = best_logreg.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"Test Accuracy of Logistic Regression: {accuracy * 100:.2f}%")

    # visualize_classical_model(best_logreg, X_test, Y_test, 'Logistic Regression')

    end = time.time()
    print(f"Time Test of logistic: {((end - start) / 60) :.2f}")


def KNN(X_test, Y_test):
    start = time.time()
    with open('KNN model.pkl', 'rb') as f:
        best_knn = pickle.load(f)

    # Make predictions on the test set
    y_pred = best_knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, y_pred)

    print(f"Test Accuracy of KNN: {accuracy * 100:.2f}%")

    # visualize_classical_model(best_knn, X_test, Y_test, 'KNN')

    end = time.time()
    print(f"Time Test of KNN: {((end - start) / 60) :.2f}")


def random_forest(X_test, Y_test):
    start = time.time()
    with open('random model.pkl', 'rb') as f:
        rand = pickle.load(f)

    # Make predictions on the test set
    y_pred = rand.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(Y_test, y_pred)
    print(f"Test Accuracy of Random Forest: {accuracy * 100:.2f}%")

    # visualize_classical_model(rand, X_test, Y_test, 'Random Forest')

    end = time.time()
    print(f"Time Test of Random Forest: {((end - start) / 60) :.2f}")


def testing_classical(image_paths_val, labels_val, type_model):
    # data preparation for classical models
    hog_features_val, val_labels = data_preparation(image_paths_val, labels_val, augment=False, type_model=type_model)

    # Flatten the HOG features
    hog_features_val_flat = hog_features_val.reshape(hog_features_val.shape[0], -1)

    SVM(hog_features_val_flat, val_labels)
    logistic_regression(hog_features_val_flat, val_labels)
    KNN(hog_features_val_flat, val_labels)
    random_forest(hog_features_val_flat, val_labels)


def visualize_classical_model(model, X_test, y_test, model_name):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Print classification report
    print(f"\n--------------------------------------------------------------{model_name}----------"
          f"-----------------------------------------------------------------------")
    print(f"Classification Report - {model_name}")
    print(classification_report(y_test, y_pred))


# don't need use Pca
def test_cnn_model(model_path, x_test, y_test, num_classes, height, width, num_channels):
    start = time.time()
    print(f"\n--------------------------------------------------------------CNN Model----------"
          f"-----------------------------------------------------------------------")
    np.random.seed(42)
    tf.random.set_seed(42)
    torch.manual_seed(42)

    # Create the CNN model
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

    # Load the model weights
    model.load_weights(model_path + '_weights.h5')

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Load the training history
    with open(model_path + '_history.pkl', 'rb') as history_file:
        history_data = joblib.load(history_file)

    acc = history_data['accuracy']
    val_acc = history_data['val_accuracy']
    loss = history_data['loss']
    val_loss = history_data['val_loss']

    # Evaluate the model on test data
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

    # Get predictions on test data
    '''
    probabilities = model.predict(x_test)
    predictions = np.argmax(probabilities, axis=-1)

    # Plot training and validation accuracy
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Show the plot
    plt.show()
'''
    print(f"Final Training Accuracy: {acc[-1] * 100:.2f}%")
    print(f"Final Validation Accuracy: {val_acc[-1] * 100:.2f}%")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    end = time.time()
    print(f"Time Test of CNN: {((end - start) / 60) :.2f}")

    '''# Display confusion matrix
    # target_names = [str(i) for i in range(1, num_classes + 1)]
    target_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                    '19', '20']
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - CNN')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()'''

    '''# Generate the classification report
    report = classification_report(y_test, predictions)
    print("\nClassification Report - CNN")
    print(report)'''


#############################################################################################
def read_test(dataset_path):
    image_paths = []
    labels = []
    y_label = []
    # Loop through each folder in the unseen test folder
    for filename in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, filename)
        folder_path = folder_path.replace('\\', '/')
        all_images = os.listdir(folder_path)

        # Loop through each file in the inner folder
        for image_name in all_images:
            image_path = os.path.join(dataset_path, filename, image_name)
            image_path = image_path.replace('\\', '/')
            # print("image path: ", image_path)
            image_paths.append(image_path)

            # label of each image
            path_elements = folder_path.strip().split('/')  # Adjust the separator based on your paths
            label = path_elements[-1]
            y_label.append(int(label))
            # print("yy", y_label)
    return image_paths, y_label


# read data
image_paths_val, labels_val = read_test(data_path)

# Classical Models (SVM ,LogisticRegression, KNN, RandomForest)
testing_classical(image_paths_val, labels_val, 'classical')

# CNN Model (using augmentation)
# Testing the model using the saved model and evaluation metrics
images_val, val_labels = data_preparation(image_paths_val, labels_val, augment=False, type_model='cnn')

model_path = './saved_model/model.h5'
test_cnn_model(model_path, images_val, val_labels, 21, 224, 224, 3)

end_time_test = time.time()
time_train = (end_time_test - start_time_test) / 60
print(f"Time of test: {time_train :.2f}")

# Reset the warning filter to its default behavior
warnings.resetwarnings()
