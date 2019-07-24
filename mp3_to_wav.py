from os import path
from pydub import AudioSegment

# files
src = "sounds/FURELISE.mp3"
dst = "sounds/test3.wav"

# convert wav to mp3
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav", bitrate="16000")