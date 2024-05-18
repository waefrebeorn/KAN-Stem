import datetime
from tensorflow.keras.callbacks import TensorBoard
from utils import load_and_preprocess_dataset  # Use absolute import
from model import create_model  # Use absolute import

def train_model(dataset_path):
    dataset = load_and_preprocess_dataset(dataset_path)
    model = create_model()

    log_dir = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(dataset, epochs=10, callbacks=[tensorboard_callback])
    return model
