import sys, subprocess, shutil
import torch, transformers, librosa, soundfile, numpy
print("python", sys.version.replace("\n", " "))
for m in (torch, transformers, librosa, soundfile, numpy):
    print(m.__name__, m.__version__)
print("libsndfile", soundfile.__libsndfile_version__)
import soxr; print("soxr", soxr.__version__)
print("cuda available", torch.cuda.is_available(), torch.version.cuda,
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
for t in ("ffmpeg", "ffprobe"):
    print(t, shutil.which(t), subprocess.run([t, "-version"], capture_output=True, text=True).stdout.splitlines()[0])
from transformers import Wav2Vec2FeatureExtractor
fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-1b")
print("do_normalize =", fe.do_normalize)
print(fe)
