import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# Set a seed value for reproducibility
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Configure a new global TensorFlow session - This is likely not necessary in TF2
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)  # Removed in TF2
# Instead of using set_session, try tf.config.threading.set_inter_op_parallelism_threads and tf.config.threading.set_intra_op_parallelism_threads to control parallelism
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
# Set file paths
# Set file paths
data_path = "/content/chest_xray"
train_path = os.path.join(data_path, "train/")
val_path = os.path.join(data_path, "val/")
test_path = os.path.join(data_path, "test/")
# Hyperparameters
hyper_dimension = 64
hyper_batch_size = 128
hyper_epochs = 1
hyper_channels = 1
hyper_mode = 'grayscale'

# Data augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255.0, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Data generators
train_generator = train_datagen.flow_from_directory(directory=train_path, target_size=(hyper_dimension, hyper_dimension),
                                                    batch_size=hyper_batch_size, color_mode=hyper_mode, class_mode='binary', seed=42)
val_generator = val_datagen.flow_from_directory(directory=val_path, target_size=(hyper_dimension, hyper_dimension),
                                                batch_size=hyper_batch_size, class_mode='binary', color_mode=hyper_mode, shuffle=False, seed=42)
test_generator = test_datagen.flow_from_directory(directory=test_path, target_size=(hyper_dimension, hyper_dimension),
                                                  batch_size=hyper_batch_size, class_mode='binary', color_mode=hyper_mode, shuffle=False, seed=42)

test_generator.reset()

# Build CNN model
cnn = Sequential([
    InputLayer(input_shape=(hyper_dimension, hyper_dimension, hyper_channels)),
    Conv2D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
cnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[AUC()])

# Train model
cnn_model = cnn.fit(train_generator, steps_per_epoch=len(train_generator), epochs=hyper_epochs,
                    validation_data=val_generator, validation_steps=len(val_generator), verbose=2)

def create_charts(model, history):
    """Generate performance charts and summary statistics."""
    auc_key = None
    val_auc_key = None
    # Iterate over the keys to find the AUC keys dynamically
    for key in history.history:
        if key.startswith('auc'):
            auc_key = key
        elif key.startswith('val_auc'):
            val_auc_key = key
    train_loss, val_loss = history.history['loss'], history.history['val_loss']
    train_auc, val_auc = history.history[auc_key], history.history[val_auc_key]
    y_true = test_generator.classes
    y_pred_prob = model.predict(test_generator, steps=len(test_generator)).T[0]
    y_pred = (y_pred_prob > 0.5).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Training vs Validation Loss
    axes[0, 0].plot(train_loss, label='Training Loss')
    axes[0, 0].plot(val_loss, label='Validation Loss')
    axes[0, 0].set_title("Training vs Validation Loss")
    axes[0, 0].legend()

    # Training vs Validation AUC
    axes[0, 1].plot(train_auc, label='Training AUC')
    axes[0, 1].plot(val_auc, label='Validation AUC')
    axes[0, 1].set_title("Training vs Validation AUC")
    axes[0, 1].legend()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'], ax=axes[1, 0])
    axes[1, 0].set_title("Confusion Matrix")
    axes[1, 0].set_xlabel("Predicted")
    axes[1, 0].set_ylabel("Actual")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    axes[1, 1].plot([0, 1], [0, 1], 'k--', label="Random (AUC = 50%)")
    axes[1, 1].plot(fpr, tpr, label=f'CNN (AUC = {auc_score*100:.2f}%)')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Summary Statistics
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / np.sum(cm)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f'[Summary Statistics]\nAccuracy = {accuracy:.2%} | Precision = {precision:.2%} | Recall = {recall:.2%} | Specificity = {specificity:.2%} | F1 Score = {f1:.2%}')

# Generate charts and summary statistics
create_charts(cnn, cnn_model)
