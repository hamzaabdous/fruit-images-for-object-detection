
# fruit-images-for-object-detection

This code appears to be a Python script for a machine learning project that involves image classification of fruits. Below is a description of the code's main components and functionality:

## 1 : Importing Libraries :

The code starts by importing necessary Python libraries such as NumPy, pandas, os, matplotlib, and OpenCV (cv2). Additionally, it imports various modules from the Keras library for building and training neural networks, as well as the train_test_split function from scikit-learn for data splitting.

## 2 : Setting Random Seed :

" np.random.seed(1) " is used to set a random seed to ensure reproducibility of the results.

## 3 : Processing Training Data :

It reads and processes training data from a directory specified by train_path. 
For each image file (with a '.jpg' extension) in that directory, it does the following:
  * Reads the image using OpenCV (cv2).
  * Extracts the fruit label from the image filename (assuming the filename format is 'label_xxx.jpg').
  * Resizes the image to a specified shape (200x200 pixels).
  * Appends the resized image to the train_images list.
  * Appends the one-hot encoded label to the train_labels list.
  * After processing all training images, it converts train_labels to a NumPy array.
  * It then splits the training data into training and validation sets using train_test_split.

## 4 : Processing Testing Data :

* Similar to training data processing, it reads and processes testing data from a directory specified by test_path. However, it does not use the labels for testing.
* The processed testing images are stored in the test_images array.

## 5 : Visualizing Training Data :

* It displays two training images with their corresponding labels.

## 6 : Creating a Convolutional Neural Network (CNN) Model :

It defines a Sequential model for a CNN with the following layers:
* Three convolutional layers with tanh activation functions.
* Two max-pooling layers.
* A flattening layer to convert the 2D feature maps into a 1D vector.
* Two dense (fully connected) layers with ReLU activation functions.
* The output layer with a softmax activation function having four units (assuming four fruit classes).

## 7 : Compiling the Model :

The model is compiled with categorical cross-entropy as the loss function, accuracy as the evaluation metric, and the Adam optimizer.

## 8 : Training the Model :

The model is trained on the training data (x_train and y_train) for 50 epochs with a batch size of 50. Validation data (x_val and y_val) is used for validation during training.

## 9 : Plotting Training History :

The code plots the training and validation accuracy as well as the training and validation loss over epochs to visualize the training progress.

## 10 : Evaluating the Model :

The model is evaluated on the validation data (x_val and y_val), and the evaluation results (loss and accuracy) are printed

## 11 : Testing the Model :

It selects a test image from the test_images_hamza array and predicts its class using the trained model. The predicted class is printed.

## 12 : Saving the Model :

The trained model is saved to a file named 'fruit-images-for-object-detection-v1.h5'.

## 13 : Loading the Model :

The code demonstrates how to load a saved model from the file and displays its summary.

## 14 : Testing the Loaded Model :

It selects a test image and predicts its class using the loaded model to verify that the loading process was successful.

This code seems to be part of a computer vision project that involves training a CNN to classify fruit images into different categories and saving the trained model for future use.

