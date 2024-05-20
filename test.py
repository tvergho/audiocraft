from audiocraft.solvers.musicgen import MusicGenSolver
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen
import torch
import torchaudio

melody, sr = torchaudio.load('/home/tyler/f0042zm/slakh/slakh2100_flac_redux/test/Track01876/stems/S00.flac')

solver = MusicGenSolver.get_eval_solver_from_sig('2dc43270', device='cuda', batch_size=1)
lm = solver.model
print(solver.model)

model = MusicGen.get_pretrained('facebook/musicgen-small')
model.lm = lm


# dataloaders = solver.dataloaders
# dataloader_train = dataloaders['train']
# for batch in dataloader_train:
#     print(batch)
#     break
# descriptions = ['lilyflower', 'happy rock', 'energetic EDM', 'sad jazz']
descriptions = ['Electric Bass music']
# wav = model.generate_with_chroma(descriptions, melody[None], sr)
wav = model.generate(descriptions)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)