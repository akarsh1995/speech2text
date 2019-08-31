import numpy as np
import shlex
import subprocess
import sys
import wave
import sounddevice as sd

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

# Define the sample rate for audio


class RecordOrRead:
    SAMPLE_RATE = 16000  # Sample rate
    duration = 5  # seconds
    file_path = None

    def __init__(self, file_path=None):
        self.file_path = file_path if isinstance(file_path, str) else None

    def read_file(self) -> np.ndarray:
        with wave.open(self.file_path, 'rb') as fin:
            fin.getframerate()
            frame_rate = fin.getframerate()
            if frame_rate != self.SAMPLE_RATE:
                print(
                    'Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic '
                    'speech recognition.'.format(frame_rate, self.SAMPLE_RATE), file=sys.stderr)
                frame_rate, audio = self.convert_samplerate()
            else:
                audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
            audio_length = fin.getnframes() * (1 / self.SAMPLE_RATE)
        return audio

    def record_audio(self):
        my_recording = sd.rec(int(self.duration * self.SAMPLE_RATE), samplerate=self.SAMPLE_RATE, channels=1)
        sd.wait()
        return my_recording

    def convert_samplerate(self):
        sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little ' \
                  '--compression 0.0 --no-dither - '.format(
            quote(self.file_path), self.SAMPLE_RATE)
        try:
            output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
        except OSError as e:
            raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(self.SAMPLE_RATE, e.strerror))

        return self.SAMPLE_RATE, np.frombuffer(output, np.int16)
