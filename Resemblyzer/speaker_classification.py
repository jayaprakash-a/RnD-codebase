from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils_modified import *
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os, glob


def preprocess_wavs():
	# Gather the wavs
	path = "/home/data/ToysFromTrash/audio/"
	wav_fpaths = list(glob.glob(os.path.join(path, '*.wav')))

	speakers = list(map(lambda wav_fpath: wav_fpath.parent.stem, wav_fpaths))
	wavs = np.array(list(map(preprocess_wav, tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths)))))
	speaker_wavs = {speaker: wavs[list(indices)] for speaker, indices in 
	                groupby(range(len(wavs)), lambda i: speakers[i])}

	np.save('preprocess_wav_files.npy', wavs)

def loadFiles():
	wavs = np.load('preprocess_wav_files.npy', allow_pickle=True)
	## Compute the embeddings
	encoder = VoiceEncoder()
	#utterance_embeds = np.array(list(map(encoder.embed_utterance, wavs)))

	#np.save('utterance_embeds.npy', utterance_embeds)
	wavs = np.load('utterance_embeds.npy', allow_pickle=True)

def plot():
	## Project the embeddings in 2D space
	plot_projections(utterance_embeds, title="Embedding projections")
	fig1 = plt.gcf()
	plt.show()
	plt.draw()
	fig1.savefig('plot.png', dpi=100)
	#plt.show()
