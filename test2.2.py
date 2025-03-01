import os
import librosa
import librosa.effects as effects
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt

def augment_audio(audio, sr):
    # 随机时间拉伸
    rate = np.random.uniform(0.8, 1.2)
    audio_stretched = effects.time_stretch(audio, rate=rate)
    
    # 随机音高调整
    n_steps = np.random.randint(-3, 3)
    audio_pitched = effects.pitch_shift(audio_stretched, sr=sr, n_steps=n_steps)
    
    return audio_pitched

# 改进特征提取：增加更多音频特征
def extract_enhanced_features(file_path, augment=False):
    try:
        audio, sr = librosa.load(file_path)
        if augment:
            audio = augment_audio(audio, sr)
        features = []
        
        # MFCCs（保留时间序列均值）
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        features.extend(mfccs_mean)
        
        # 色度特征
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        features.extend(chroma_mean)
        
        # 频谱对比度
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        features.extend(contrast_mean)
        
        # 过零率
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)
        features.append(zcr_mean)
        
        # 均方根能量
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)
        features.append(rms_mean)
        
        # 谱质心
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        centroid_mean = np.mean(centroid)
        features.append(centroid_mean)
        
        return np.array(features)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# 路径配置
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'genres')
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# 提取特征和标签，增加数据增强
features, labels = [], []
for genre in genres:
    genre_path = os.path.join(data_path, genre)
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        # 原始数据特征提取
        feat = extract_enhanced_features(file_path)
        if feat is not None:
            features.append(feat)
            labels.append(genre)
        # 增强数据特征提取
        feat_augmented = extract_enhanced_features(file_path, augment=True)
        if feat_augmented is not None:
            features.append(feat_augmented)
            labels.append(genre)

# 转换为数组并编码标签
X = np.array(features)
y = LabelEncoder().fit_transform(labels)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

# 构建神经网络模型
model = Sequential([
    Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.6),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(len(genres), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 早停和学习率衰减
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

history = model.fit(
    X_train, y_train_onehot,
    epochs=300,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# 评估模型
y_pred = model.predict(X_test).argmax(axis=1)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# 分类报告和混淆矩阵
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=genres))

plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=genres, yticklabels=genres)
plt.title('Confusion Matrix (Neural Network)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()