import tensorflow as tf

def load_and_preprocess_dataset(dataset_path):
    dataset = tf.data.Dataset.list_files(dataset_path + '/*.wav')
    dataset = dataset.map(lambda x: tf.audio.decode_wav(tf.io.read_file(x))[0])
    dataset = dataset.batch(32)
    return dataset
