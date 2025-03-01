import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import audiomentations as aug

genres = ['blues','classical','country','disco','hiphop',
              'jazz','metal','pop','reggae','rock']

# ================== 配置参数 ==================
SR = 22050          # 采样率
DURATION = 30       # 音频长度（秒）
N_MELS = 128        # 梅尔带数
N_FFT = 2048        # FFT窗口大小
HOP_LENGTH = 512    # 帧移
NUM_CLASSES = 10    # 类别数

# ================== 数据增强管道 ==================
augmenter = aug.Compose([
    aug.TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    aug.PitchShift(min_semitones=-3, max_semitones=3, p=0.5),
    aug.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3)
])

# ================== 特征提取函数 ==================
def audio_to_melspectrogram(audio, sr=SR):
    # 数据增强（仅在训练时应用）
    if tf.random.uniform(()) > 0.5:
        audio = augmenter(audio, sample_rate=sr)
    
    # 确保音频长度固定
    if len(audio) < sr * DURATION:
        audio = np.pad(audio, (0, max(0, sr * DURATION - len(audio))))
    else:
        audio = audio[:sr * DURATION]
    
    # 生成梅尔频谱图
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 标准化到[-1,1]范围
    mel_spec_db = (mel_spec_db + 80) / 80
    return mel_spec_db[..., np.newaxis]  # 添加通道维度

# ================== 数据加载管道 ==================
def load_and_process_data(file_path, label):
    # 加载音频
    audio, _ = librosa.load(file_path, sr=SR, duration=DURATION)
    
    # 标准化音频长度
    if len(audio) < SR*DURATION:
        audio = np.pad(audio, (0, max(0, SR*DURATION - len(audio))))
    else:
        audio = audio[:SR*DURATION]
    
    # 转换为频谱图
    spec = tf.numpy_function(audio_to_melspectrogram, [audio], tf.float32)
    spec.set_shape((N_MELS, (SR*DURATION)//HOP_LENGTH + 1, 1))
    return spec, label

# ================== 构建数据集 ==================
def build_dataset(data_dir):
    file_paths = []
    labels = []
    for i, genre in enumerate(genres):
        genre_dir = os.path.join(data_dir, genre)
        for fn in os.listdir(genre_dir):
            file_paths.append(os.path.join(genre_dir, fn))
            labels.append(i)
    
    # 创建TensorFlow Dataset
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.shuffle(2000, reshuffle_each_iteration=False)
    ds = ds.map(lambda x,y: tf.py_function(
        load_and_process_data, [x,y], (tf.float32, tf.int32)),
        num_parallel_calls=tf.data.AUTOTUNE)
    return ds

# ================== 构建CNN模型 ==================
def build_cnn_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.2),
        
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),
        
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.4),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ================== 主程序 ==================
if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), 'genres')
    full_ds = build_dataset(data_dir)
    
    train_size = int(0.8 * len(full_ds))
    train_ds = full_ds.take(train_size).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = full_ds.skip(train_size).batch(32).prefetch(tf.data.AUTOTUNE)
    
    input_shape = (N_MELS, (SR*DURATION)//HOP_LENGTH + 1, 1)
    model = build_cnn_model(input_shape)
    
    callbacks = [
        callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=callbacks
    )
    
    test_ds = val_ds.unbatch().batch(32)
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred = model.predict(test_ds).argmax(axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=genres))
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(confusion_matrix(y_true, y_pred), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=genres, yticklabels=genres)
    plt.title('Optimized CNN Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()