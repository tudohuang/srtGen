import whisper
from pydub import AudioSegment
import numpy as np
import opencc
import torch
import re

def ms_to_srt_time(ms):
    s = ms // 1000
    ms = ms % 1000
    m = s // 60
    s = s % 60
    h = m // 60
    m = m % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def generate_srt(file_path):
    # Load the Whisper model
    model = whisper.load_model("medium", device="cuda")
    
    # Read the audio file
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    # Initialize OpenCC for Chinese conversion (Simplified to Traditional)
    converter = opencc.OpenCC('s2t')
    
    # Define regex to split sentences
    sentence_splitter = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
    
    # Initialize SRT subtitle list
    srt_subs = []
    
    # Split audio and perform transcription
    start_time = 0
    for sentence in re.split(sentence_splitter, audio.raw_data.decode("latin1")):
        if not sentence.strip():
            continue
        
        # Convert sentence to audio
        segment = AudioSegment.from_file(file_path)
        sentence_duration_ms = len(segment)
        audio_data = np.array(segment.get_array_of_samples(), dtype=np.float32)
        
        if segment.channels == 2:
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
        
        # Normalize audio data
        audio_data = audio_data / 32768  # Adjust the range to [-1.0, 1.0]
        audio_data = audio_data.astype(np.float32)
        audio_data = torch.from_numpy(audio_data).to("cuda")
        
        # Transcribe audio to text
        result = model.transcribe(audio_data)
        traditional_text = converter.convert(result['text'])
        
        srt_subs.append({
            'start_time': start_time,
            'end_time': start_time + sentence_duration_ms,
            'text': traditional_text
        })
        
        start_time += sentence_duration_ms
    
    # Write SRT file
    with open('output.srt', 'w', encoding='utf-8') as f:
        for i, sub in enumerate(srt_subs):
            f.write(f"{i + 1}\n")
            f.write(f"{ms_to_srt_time(sub['start_time'])} --> {ms_to_srt_time(sub['end_time'])}\n")
            f.write(f"{sub['text']}\n\n")

