import os
import random
import numpy as np

import json
import soundfile as sf

BLOCK_SIZE = 500
SAMPLE_RATE = 16000
SAMPLE_LENGTH = 4

INSTRUMENT_NAMES = [
        "bass",
        "brass",
        "flute",
        "guitar",
        "keyboard",
        "mallet",
        "organ",
        "reed",
        "string",
        "synth_lead",
        "vocal",
        ]

TIMBRES = [
        "acoustic",
        "electronic",
        "synthetic",
        ]

QUALITIES = [
        "bright",
        "dark",
        "distortion",
        "fast_decay",
        "long_release",
        "multiphonic",
        "nonlinear_env",
        "percussive",
        "reverb",
        "tempo-synced",
        ]


def slide_fft(data, bs=BLOCK_SIZE):
    res = []
    for i in range(0, len(data), bs):
        slide = data[i:i+bs]
        if len(slide) < bs:
            continue
        res.append(np.fft.fft(slide))
    return np.array(res)


def slide_ifft(data, bs=BLOCK_SIZE):
    res = []
    for slide in data:
        res.append(np.fft.ifft(slide))
    return np.array(res).reshape(-1)


class Sound:
    def __init__(self, sound_id, path, pitch, velocity, family, timbre, quality):
        self.sound_id = sound_id
        self.path = path
        self.pitch = pitch
        self.velocity = velocity
        self.family = family
        self.timbre = timbre
        self.quality = quality
    def __str__(self):
        return "Sound(id={}, path=.../{} pitch={}, velocity={}, family={}, timbre={}, quality={})".format(
                repr(self.sound_id),
                os.path.basename(self.path),
                self.pitch,
                self.velocity,
                INSTRUMENT_NAMES[self.family],
                TIMBRES[self.timbre],
                str([QUALITIES[x] + "(" + str(x) + ")" for x in self.quality])
                )

    def get_sound(self):
        sound, sample_rate = sf.read(self.path)
        assert len(sound) == SAMPLE_RATE * SAMPLE_LENGTH

        return sound

def spectrums(sounds):
    freqs = np.array([slide_fft(x) for x in sounds])

    re_pos, re_neg, im_pos, im_neg = (
        np.real(freqs).reshape(freqs.shape[0], 1, *freqs.shape[1:]).clip(0, None),
        -np.real(freqs).reshape(freqs.shape[0], 1, *freqs.shape[1:]).clip(None, 0),
        np.imag(freqs).reshape(freqs.shape[0], 1, *freqs.shape[1:]).clip(0, None),
        -np.imag(freqs).reshape(freqs.shape[0], 1, *freqs.shape[1:]).clip(None, 0),
    )

    inp = np.log(np.concatenate([re_pos, re_neg, im_pos, im_neg], axis=1) + 1)
    return inp

def sounds(spectrums):
    delog = np.exp(spectrums) - 1
    fourier = delog[:,0] - delog[:,1] + 1j * delog[:,2] - 1j * delog[:,3]

    result = np.array([slide_ifft(x) for x in fourier])
    return result

def load_data(path=os.path.expanduser("~/nsynth/"), amount=-1, silent=False):
    if not silent:
        print("...")

    json_data = json.load(open(os.path.join(path, "examples.json")))

    data = []


    i = 0
    for sound_id, props in json_data.items():
        if not silent:
            if amount > 0:
                print("\033[1A[{}/{}]".format(i, amount))
            else:
                print("\033[1A[{}/{}]".format(i, len(json_data)))


        i += 1
        filename = os.path.join(path, "audio", sound_id + ".wav")

        assert props["sample_rate"] == SAMPLE_RATE

        data.append(Sound(
            sound_id,
            filename,
            props["pitch"],
            props["velocity"],
            props["instrument_family"],
            props["instrument_source"],
            [x for x in range(len(QUALITIES)) if props["qualities"][x]]
            ))

        if i >= amount and amount > 0:
            break

    random.shuffle(data)
    return data

if __name__ == "__main__":
    import sounddevice as sd
    data, sample_rate, sample_length = load_data(amount=100)
    print("Sample rate = {}, sample length = {}".format(sample_rate, sample_length))
    random.shuffle(data)

    for sound in data[:20]:
        print(str(sound))
        sd.play(sound.get_sound(), sample_rate, blocking=True)
