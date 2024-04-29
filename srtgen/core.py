import whisper
from pydub import AudioSegment
import numpy as np
import opencc
import torch
import re
from tqdm import tqdm

def audio_to_srt(audio_file_path, output_srt_path, model_name="medium", language="s2t"):
    # 加載模型 + GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name, device=device)

    # 讀取音頻文件
    audio = AudioSegment.from_file(audio_file_path).set_frame_rate(16000).set_channels(1)

    # 初始化 OpenCC 進行繁體和簡體中文的轉換
    converter = opencc.OpenCC(language)

    # 定義斷句的正則表達式
    sentence_splitter = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

    # 切割音頻並進行語音轉文字
    sentences = re.split(sentence_splitter, audio.raw_data.decode("latin1"))
    srt_subs = []
    start_time = 0

    for sentence in tqdm(sentences):
        # 計算句子的持續時間
        segment_duration_ms = len(sentence) * (1000 / audio.frame_rate)

        if not sentence.strip():  # 跳過空白句子
            start_time += segment_duration_ms
            continue

        # 從音頻中切割出對應句子的片段
        segment = audio[start_time:start_time + segment_duration_ms]
        start_time += segment_duration_ms  # 更新起始時間

        # 將片段的音頻數據轉換為模型所需格式
        audio_data = np.array(segment.get_array_of_samples(), dtype=np.float32)
        if segment.channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
        audio_data = (audio_data / 32768).astype(np.float32)  # 轉換數據範圍至 [-1.0, 1.0]
        audio_data = torch.from_numpy(audio_data).to(device)  # 轉移到 GPU

        # 進行語音轉文字
        result = model.transcribe(audio_data)
        text = converter.convert(result['text'])  # 轉換為指定語言的文字

        # 生成 SRT 字幕
        srt_subs.append({
            'start_time': start_time - segment_duration_ms,
            'end_time': start_time,
            'text': text
        })

    # 寫入 SRT 檔案
    with open(output_srt_path, 'w', encoding='utf-8') as f:
        for i, sub in enumerate(srt_subs):
            f.write(f"{i+1}\n")
            f.write(f"{ms_to_srt_time(sub['start_time'])} --> {ms_to_srt_time(sub['end_time'])}\n")
            f.write(f"{sub['text']}\n\n")

def ms_to_srt_time(ms):
    s = int(ms // 1000)
    ms = int(ms % 1000)
    m = s // 60
    s = s % 60
    h = m // 60
    m = m % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

