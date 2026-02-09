import wave
import numpy as np
from scipy.signal import resample

def read_wav(file_path, target_fs):   

    # ------- ĐỌC WAV GỐC -------
    with wave.open(file_path, 'rb') as wav:
        fs_original = wav.getframerate()
        n_frames    = wav.getnframes()
        n_channels  = wav.getnchannels()
        sampwidth   = wav.getsampwidth()

        raw = wav.readframes(n_frames)

    # ------- CHUYỂN DỮ LIỆU SỐ -------
    if sampwidth == 1:
        audio = np.frombuffer(raw, dtype=np.uint8) - 128
    elif sampwidth == 2:
        audio = np.frombuffer(raw, dtype=np.int16)
    elif sampwidth == 3:
        audio = np.frombuffer(raw, dtype=np.int32) >> 8
    else:
        raise ValueError("Format WAV không hỗ trợ")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)

    # ------- RESAMPLE BẮT BUỘC VỀ 480 kHz -------
    if fs_original != target_fs:
        N_new = int(len(audio) * target_fs / fs_original)
        audio = resample(audio, N_new)
    
    return audio.astype(np.float32), target_fs 
