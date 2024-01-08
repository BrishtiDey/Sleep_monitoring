from gtts import gTTS
from IPython.display import Audio
from IPython.display import display

for voice in ["Wake up!, Wake up!"]: 
            tts = gTTS(format(voice)) 
            tts.save('1.wav')
