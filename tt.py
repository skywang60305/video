import librosa

file_path = r'c:\Users\10603\Desktop\test\genres\blues\blues.00000.au'
try:
    audio, sr = librosa.load(file_path)
    print(f"Successfully loaded {file_path}")
except Exception as e:
    print(f"Error loading {file_path}: {e}")