from deepspeech_model.deepspeech import SpeechToText

if __name__ == '__main__':
    s = SpeechToText()
    print("output", s.predict('sample.wav'))
