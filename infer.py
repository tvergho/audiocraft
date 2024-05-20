from audiocraft.solvers.musicgen import MusicGenSolver
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen
import torch

solver = MusicGenSolver.get_eval_solver_from_sig('1b0cb9c9', device='cuda')
# solver.model.cfg_coef = 3.0
# pretrained = MusicGen.get_pretrained('facebook/musicgen-small', lm=lm)
model = MusicGen(name='mymusicgen', compression_model=solver.compression_model, lm=solver.model, max_duration=30)
model.set_generation_params(cfg_coef=1.0, duration=15)

# model = MusicGen.get_pretrained('facebook/musicgen-small', lm=lm)
# model.lm = lm

descriptions = ['acoustic guitar music']
wav = model.generate(descriptions)

for idx, one_wav in enumerate(wav):
  # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
  audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)