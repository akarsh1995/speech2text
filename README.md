### **Convert audio file from speech to text (Offline mode)**

To automate the stuff download the setup.sh file from drive link https://bit.ly/2Le6ytK

Give executable permissions to the downloaded file.

```sh
chmod +x setup.sh
./setup.sh
```

This will create the directory named speech2text in the current working directory.
It will download the relevant model (2 GB) which helps converting speech to text.

The sample.wav file is provided in the root directory of the repository.  

To test run from the root directory.

```sh
python main.py
```

> output :

```
Loading model from file models/output_graph.pbmm
TensorFlow: v1.13.1-10-g3e0cc5374d
DeepSpeech: v0.5.1-0-g4b29b78
Loaded model in 0.0159s.
Loading language model from files models/lm.binary models/trie
Loaded language model in 1.8s.
Warning: original sample rate (8000) is different than 16000hz. Resampling might produce erratic speech recognition.
Running inference.
Inference took 14.607s

output (from sample.wav): not gently but wake her now the news struck doubt into the restless minds once we stood beside the shore a chunk in the wall allowed a draft to blow fastened two pins on each side the cold dip restores health and zest he takes the oath of office each march the sand drifts over the sill of the old house the point of the steel pen was bent and twisted there is a lag between thought and act
```

#### Further enhancements:

- Take the live sound input from the microphone and convert speech2text on the go.

##### PS:  

- Sample rate currently supported is 16000hz
- If sample rate is higher then script will perform downsampling. This may result in less accurate result.

