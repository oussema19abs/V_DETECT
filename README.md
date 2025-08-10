# Real-time Fight Detection in Videos

This project is a proof-of-concept for a real-time fight and violence detection system using deep learning. The model is built with TensorFlow and Keras and is designed to classify video sequences as either containing a "Fight" or "NonFight".

## 📜 Project Overview

The primary goal of this project is to develop a model capable of identifying violent actions in video footage. This is achieved by combining a convolutional neural network (CNN) for spatial feature extraction from individual frames and a recurrent neural network (RNN) to understand the temporal relationship between frames. This approach allows the model to learn the patterns of movement and interaction that are characteristic of a physical altercation.

### Key Features:

* **Dataset Combination**: Utilizes four different public datasets to create a more robust and diverse training set.
* **Transfer Learning**: Employs the MobileNetV2 architecture, pre-trained on ImageNet, to extract meaningful features from each video frame.
* **Temporal Analysis**: Uses a Bidirectional GRU (Gated Recurrent Unit) layer to analyze the sequence of frames and understand the temporal context.
* **Two-Phase Training**: Implements an initial training phase with a frozen base model followed by a fine-tuning phase to improve accuracy.

## ⚙️ Methodology

1.  **Data Preparation**:
    * Four datasets were downloaded from Kaggle: RWF-2000, Hockey Fights, Fight/No-Fight Surveillance, and Movies Fight Detection.
    * The videos from these datasets were extracted and unified into a single directory structure with `train/Fight`, `train/NonFight`, `val/Fight`, and `val/NonFight` subdirectories.

2.  **Video Preprocessing**:
    * A custom `preprocess_video` function was created to read video files, sample a fixed number of frames (`SEQUENCE_LENGTH = 60`), resize them to a consistent size (`IMG_SIZE = 112x112`), and normalize the pixel values.
    * A `tf.data.Dataset` pipeline was built to efficiently load and preprocess the video data during training, including data augmentation for the training set.

3.  **Model Architecture**:
    * The model uses a `TimeDistributed` layer to apply the MobileNetV2 feature extractor to each of the 60 frames in a sequence.
    * The output features are then passed through a `Bidirectional GRU` layer followed by another `GRU` layer to learn the temporal patterns.
    * A final `Dense` layer with a sigmoid activation function outputs a probability score indicating the likelihood of a fight.

4.  **Training and Fine-Tuning**:
    * **Initial Training**: The model was first trained for 20 epochs with the initial layers of MobileNetV2 frozen. This allows the newly added RNN and Dense layers to learn without disrupting the pre-trained weights.
    * **Fine-Tuning**: After the initial training, the entire model was unfrozen and trained for an additional 30 epochs with a lower learning rate to fine-tune the entire network for better performance.

## 📊 Results

The model was evaluated on a sample of videos from the validation set, achieving an accuracy of **90.00%**.

* **Precision (NonFight)**: 1.00
* **Recall (NonFight)**: 0.80
* **Precision (Fight)**: 0.83
* **Recall (Fight)**: 1.00

The confusion matrix on the sample data was: