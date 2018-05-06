import os
import random

import json
import soundfile as sf

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
        sound, _ = sf.read(self.path)
        return sound

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

        sample_rate = props["sample_rate"]

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

    return (data, sample_rate, len(data[0].get_sound()))

if __name__ == "__main__":
    import sounddevice as sd
    data, sample_rate, sample_length = load_data(amount=100)
    print("Sample rate = {}, sample length = {}".format(sample_rate, sample_length))
    random.shuffle(data)

    for sound in data[:20]:
        print(str(sound))
        sd.play(sound.get_sound(), sample_rate, blocking=True)
