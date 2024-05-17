import os
import librosa
import numpy as np
import tensorflow as tf
import gradio as gr

# 1. Load and preprocess audio files
def load_and_preprocess_audio(file_path, sr=44100):
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        audio = librosa.util.normalize(audio)
        return audio
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_stft_features(audio, n_fft=2048, hop_length=512):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    stft_db = librosa.amplitude_to_db(abs(stft))
    return stft_db

def prepare_dataset(dataset_path):
    data = []
    labels = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                audio = load_and_preprocess_audio(file_path)
                if audio is not None:
                    stft_features = extract_stft_features(audio)
                    data.append(stft_features)
                    labels.append(os.path.basename(root))  # Assuming folder names are labels
    return np.array(data), np.array(labels)

# 2. Define the KAN model layers
class KANLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, num_splines, grid_size):
        super(KANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_splines = num_splines
        self.grid_size = grid_size
        self.grid_points = tf.Variable(tf.linspace(0.0, 1.0, grid_size), trainable=True)
        self.splines = [self.add_weight(shape=(input_dim, grid_size), initializer='random_normal', trainable=True) for _ in range(output_dim)]

    def call(self, inputs):
        outputs = []
        for i in range(self.output_dim):
            spline_coefficients = self.splines[i]
            spline_values = tf.map_fn(lambda x: self.spline_function(x, spline_coefficients), inputs)
            outputs.append(tf.reduce_sum(spline_values, axis=1))
        return tf.stack(outputs, axis=1)

    def spline_function(self, x, coefficients):
        b_splines = tf.map_fn(lambda g: self.b_spline_basis(x, g), self.grid_points)
        return tf.reduce_sum(coefficients * b_splines, axis=1)

    def b_spline_basis(self, x, grid_point):
        return tf.maximum(0.0, 1.0 - tf.abs(x - grid_point))

class SimplifiedKANLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, num_splines):
        super(SimplifiedKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_splines = num_splines
        self.splines = [self.add_weight(shape=(input_dim,), initializer='random_normal', trainable=True) for _ in range(output_dim)]

    def call(self, inputs):
        outputs = []
        for i in range(self.output_dim):
            spline_coefficients = self.splines[i]
            spline_values = self.simplified_spline_function(inputs, spline_coefficients)
            outputs.append(tf.reduce_sum(spline_values, axis=1))
        return tf.stack(outputs, axis=1)

    def simplified_spline_function(self, x, coefficients):
        return coefficients * tf.nn.relu(x)

# 3. Define the Hybrid KAN model
class HybridKANModel(tf.keras.Model):
    def __init__(self, input_shape, kan_layers, num_classes):
        super(HybridKANModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.kan_layers = kan_layers
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        for layer in self.kan_layers:
            x = layer(x)
        return self.dense(x)

# 4. Create TensorFlow Dataset
def create_tf_dataset(data, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# Paths
dataset_path = 'path/to/your/handpicked_dataset'
validation_dataset_path = 'path/to/your/validation_dataset'
test_dataset_path = 'path/to/your/test_dataset'

# Prepare the dataset
data, labels = prepare_dataset(dataset_path)
val_data, val_labels = prepare_dataset(validation_dataset_path)
test_data, test_labels = prepare_dataset(test_dataset_path)

# Encode labels as integers
label_to_int = {label: i for i, label in enumerate(np.unique(labels))}
labels = np.array([label_to_int[label] for label in labels])
val_labels = np.array([label_to_int[label] for label in val_labels])
test_labels = np.array([label_to_int[label] for label in test_labels])

# Ensure the data has the right shape for the model
data = np.expand_dims(data, axis=-1)  # Add channel dimension if necessary
val_data = np.expand_dims(val_data, axis=-1)
test_data = np.expand_dims(test_data, axis=-1)

# Convert labels to one-hot encoding
num_classes = len(label_to_int)
labels = tf.keras.utils.to_categorical(labels, num_classes)
val_labels = tf.keras.utils.to_categorical(val_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

# Create TensorFlow Dataset
train_dataset = create_tf_dataset(data, labels)
val_dataset = create_tf_dataset(val_data, val_labels)
test_dataset = create_tf_dataset(test_data, test_labels)

# Define KAN layers
kan_layers = [KANLayer(input_dim=256, output_dim=128, num_splines=10, grid_size=50),
              SimplifiedKANLayer(input_dim=128, output_dim=64, num_splines=10)]

# Define and compile the model
input_shape = (data.shape[1], data.shape[2], 1)  # Adjust input shape as necessary
hybrid_model = HybridKANModel(input_shape=input_shape, kan_layers=kan_layers, num_classes=num_classes)
hybrid_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Add Model Checkpoints and Early Stopping
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/kan_model_{epoch}',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True)

# Train the model
hybrid_model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[checkpoint_callback, early_stopping_callback])

# Evaluate the model performance on the test dataset
test_loss, test_accuracy = hybrid_model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy:.4f}')

# Predict stems and evaluate performance
predicted_stems = hybrid_model.predict(test_dataset)

# Function to evaluate performance
def evaluate_performance(predictions, true_labels):
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(true_labels, axis=1))
    print(f'Prediction accuracy: {accuracy:.4f}')

# Evaluate the predictions
evaluate_performance(predicted_stems, test_labels)

# 7. Gradio App
def separate_stems(audio_file):
    audio, sr = librosa.load(audio_file, sr=44100)
    audio = librosa.util.normalize(audio)
    stft_features = librosa.stft(audio, n_fft=2048, hop_length=512)
    stft_db = librosa.amplitude_to_db(abs(stft_features))
    
    stft_db = np.expand_dims(stft_db, axis=-1)
    stft_db = np.expand_dims(stft_db, axis=0)
    predicted_stems = hybrid_model.predict(stft_db)
    
    return predicted_stems

# Define Gradio interface
inputs = gr.inputs.Audio(source="upload", type="filepath")
outputs = gr.outputs.Textbox()

gr.Interface(fn=separate_stems, inputs=inputs, outputs=outputs, title="KAN Stem Separation", description="Upload an audio file to separate its stems using KAN").launch()
