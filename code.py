import pyaudio
import io
import wave
from utils.splitTrainDev import *
from utils.splitTrainDev import main

def record(RECORD_SECONDS=5, WAVE_OUTPUT_FILENAME="output.wav", CHUNK=1024, FORMAT=pyaudio.paInt16, CHANNELS=2, RATE=44100, frames=[]):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,input=True, frames_per_buffer=CHUNK)
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return WAVE_OUTPUT_FILENAME

def play(path, CHUNK = 1024):

    wf = wave.open(path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
    data = wf.readframes(CHUNK)
    while data != '':
        stream.write(data)
        data = wf.readframes(CHUNK)
    stream.stop_stream()
    stream.close()
    p.terminate()

def load_model(path="models"):
    path = r'model/dementia.cpkt.data-00000-of-00001'
    return path

def predict_audio(path):
    model = load_model()
    return predict(model,path)

def save(bytes, filepath="audios/uploads/upload.wav"):

    with open(filepath,'wb') as f:
        f.write(bytes.read())
    return filepath