from deepspeech_model.deepspeech import SpeechToText

if __name__ == '__main__':
    s = SpeechToText()
    print(s.predict('sample.wav'))
