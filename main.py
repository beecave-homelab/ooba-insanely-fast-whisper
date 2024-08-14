# pipx install insanely-fast-whisper --force --pip-args="--ignore-requires-python" soundfile keyboard
import gradio as gr
import speech_recognition as sr
import torch
import numpy as np
import keyboard

import time
from transformers import pipeline, file_utils
from transformers.utils import is_flash_attn_2_available

from pydub import AudioSegment
import os
import soundfile as sf
import onnxruntime as ort 
from dotenv import load_dotenv
import shutil

ort.set_default_logger_severity(3) # remove warning

# Load environment variables from .env file
load_dotenv()

# Path to the directory where the Silero VAD model will be stored
model_dir = os.getenv("SILERO_VAD_MODEL_DIR")  # INPUT_REQUIRED {path to store Silero VAD model}

if not model_dir:
    raise ValueError("SILERO_VAD_MODEL_DIR environment variable is not set in .env file.")

# Ensure the model directory exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_file = os.path.join(model_dir, "silero_vad.onnx")

# Download the Silero VAD model if it doesn't exist
if not os.path.exists(model_file):
    print(f"Downloading Silero VAD model to {model_dir}...")
    torch.hub.download_url_to_file('https://models.silero.ai/models/vad/silero_vad.onnx', model_file)
    # Assuming the utils file is also needed
    shutil.copyfile(
        torch.hub.get_dir() + '/snakers4_silero-vad_master/files/vad_utils.py', 
        os.path.join(model_dir, 'vad_utils.py')
    )
else:
    print(f"Silero VAD model already exists in {model_dir}.")

# Import the utils from the downloaded file
import importlib.util
spec = importlib.util.spec_from_file_location("vad_utils", os.path.join(model_dir, "vad_utils.py"))
vad_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vad_utils)

# Load Silero VAD model and utilities
model = torch.jit.load(model_file)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = (
    vad_utils.get_speech_timestamps, 
    vad_utils.save_audio, 
    vad_utils.read_audio, 
    vad_utils.VADIterator, 
    vad_utils.collect_chunks
)
vad_iterator = VADIterator(model)

input_hijack = {
    'state': False,
    'value': ["", ""]
}

last_output = ""
previous_state = "silence"
buffer = np.array([])

def is_silence(audio):
    global buffer
    sample_rate, frame_data = audio # Separate the sample rate and frame data
    window_size_samples = 512 # Number of samples in a single audio chunk
    speech_chunks = 0 # Counter for chunks detected as speech

    for i in range(0, len(frame_data), window_size_samples):
        chunk = frame_data[i: i+window_size_samples]
        buffer = np.concatenate((buffer[-2*window_size_samples:], chunk)) # Keep only the last 3 chunks in the buffer
        if len(buffer) < 3*window_size_samples: # If less than 3 chunks in buffer, continue to next iteration
            continue

        buffer_tensor = torch.from_numpy(buffer).float() # Convert buffer to a PyTorch tensor and then to float
        speech_prob = model(buffer_tensor, sample_rate).item() # Calculate speech probability for the buffer
        if speech_prob > 0.9:
            speech_chunks += 1
            if speech_chunks >= 2:
                return False

    vad_iterator.reset_states()
    return True

# insanely-fast-whisper, openai/whisper-large-v3, mps for Mac devices
pipe = pipeline("automatic-speech-recognition", model="distil-whisper/distil-large-v2", torch_dtype=torch.float16, device="cuda:0", model_kwargs={"use_flash_attention_2": is_flash_attn_2_available()},)

if not is_flash_attn_2_available():
    print("Flash Attention 2.0 is not available. Using bettertransformer instead.") # enable flash attention through pytorch sdpa
    pipe.model = pipe.model.to('cuda')

def insanely_fast_whisper(audio_data):
    global last_output
    start_time = time.time()
    
    output = pipe(
        audio_data,
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    last_output = output['text']
    return output['text']

directory = os.path.dirname(os.path.realpath(__file__))
filename_voice = os.path.join(directory, 'temp_voice.wav')

def do_stt(audio):
    transcription = ""
    r = sr.Recognizer()

    # Convert to AudioData, Obtain the frame data as a NumPy array
    audio_data = sr.AudioData(sample_rate=audio[0], frame_data=audio[1], sample_width=4)
    new_data = np.frombuffer(audio_data.frame_data, dtype=np.int16)
    
    if os.path.exists(filename_voice):
        old_data, old_sample_rate = sf.read(filename_voice, dtype='int16')
        combined_data = np.concatenate((old_data, new_data)) # Concatenate the old data with the new data
        sf.write(filename_voice, combined_data, audio_data.sample_rate)
    else:
        sf.write(filename_voice, new_data, audio_data.sample_rate)
    
    transcription = insanely_fast_whisper(filename_voice)
    return transcription

def generate_transcribe():
    keyboard.send("enter")

def auto_transcribe(file_path, auto_submit):
    global last_output, previous_state
    audio, sample_rate = sf.read(file_path)
    audio = (sample_rate, audio)

    if not is_silence(audio):
        transcription = do_stt(audio)
        previous_state = "talking"
    else:
        transcription = ""
        if previous_state == "talking":
            generate_transcribe()
            transcription = last_output
        last_output = ""
        previous_state = "silence"
        sf.write(filename_voice, audio[1], audio[0])
    if auto_submit:
        input_hijack.update({"state": True, "value": [transcription, transcription]})
    return transcription, None

def ui():
    with gr.Blocks() as demo:
        with gr.Row():
            upload_button = gr.UploadButton(label="Upload Audio File", file_types=["audio"], file_count="single")
            auto_submit = gr.Checkbox(label='Submit the transcribed audio automatically', value=True, visible=False)
            text_box = gr.Textbox(label="Transcription Output")

        upload_button.upload(
            auto_transcribe, [upload_button, auto_submit], [text_box]
        ).then(
            None, auto_submit, None, _js="(False) => { console.log('Check:', check); if (check) { document.getElementById('Generate').click(); }}"
        )
        
        demo.launch()

# Launch the Gradio interface
ui()