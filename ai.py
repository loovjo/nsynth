import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

import dataloader

DATA = dataloader.load_data()

BATCH_SIZE = 10

SAVE_PATH = LOAD_PATH = "ai.pt"

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv_1 = nn.Conv2d(4, 50, 5)
        self.conv_2 = nn.Conv2d(50, 1, 5)

    def forward(self, x):
        x = self.conv_1(x)
        x = nn.MaxPool2d(4)(x)
        x = nn.functional.relu(x)
        x = self.conv_2(x)
        x = nn.MaxPool2d(3)(x)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.deconv_2 = nn.ConvTranspose2d(1, 50, 5)
        self.deconv_3 = nn.ConvTranspose2d(50, 4, 5)

    def forward(self, x):
        x = nn.Upsample(scale_factor=3)(x)
        x = self.deconv_2(x)
        x = nn.Upsample(scale_factor=4)(x)
        x = self.deconv_3(x)
        x = nn.functional.relu(x)

        return x


def test_show(amount=4):
    import sounddevice as sd
    import matplotlib.pyplot as plt


    sounds = [x.get_sound() for x in DATA[:amount]]

    freqs = np.array([dataloader.slide_fft(x) for x in sounds])
    re_pos, re_neg, im_pos, im_neg = (
        np.real(freqs).reshape(freqs.shape[0], 1, *freqs.shape[1:]).clip(0, None),
        -np.real(freqs).reshape(freqs.shape[0], 1, *freqs.shape[1:]).clip(None, 0),
        np.imag(freqs).reshape(freqs.shape[0], 1, *freqs.shape[1:]).clip(0, None),
        -np.imag(freqs).reshape(freqs.shape[0], 1, *freqs.shape[1:]).clip(None, 0),
    )

    inp = np.log(np.concatenate([re_pos, re_neg, im_pos, im_neg], axis=1) + 1)

    encoded = enc(Variable(torch.Tensor(inp)))
    decoded = dec(encoded)

    print("Despectruming")
    decoded_sound = np.real(dataloader.sounds(decoded.data.numpy()))

    print("Done")

    for i, x in enumerate(decoded):
        for j in range(4):
            plt.subplot(2, 5, j + 1)
            plt.imshow(inp[i, j])

        plt.subplot(2, 5, 5)
        plt.plot(sounds[i])

        for j in range(4):
            plt.subplot(2, 5, j + 6)
            plt.imshow(decoded.data[i, j])

        plt.subplot(2, 5, 10)
        plt.plot(decoded_sound[i])

        plt.show()

        sd.play(sounds[i], blocking=True)
        sd.play(decoded_sound[i], blocking=True)

def save():
    save_dict = {
            "enc": enc.state_dict(),
            "enc_opt": enc_opt.state_dict(),
            "dec": dec.state_dict(),
            "dec_opt": dec_opt.state_dict(),
            "epoch": EPOCH,
            "train_loss_history": train_loss_history,
            }

    torch.save(save_dict, SAVE_PATH)

def load_all():
    global enc, dec, enc_opt, dec_opt, EPOCH, train_loss_history

    enc = Encoder()
    dec = Decoder()

    enc_opt = optim.SGD(enc.parameters(), lr=0.1, momentum=0.4)
    dec_opt = optim.SGD(dec.parameters(), lr=0.1, momentum=0.4)
    EPOCH = 0
    train_loss_history = []


    print("Load old state? [Y/n] ", flush=True, end="")

    if os.path.isfile(LOAD_PATH) and input().lower() != "n":
        print("Loading old state")
        state = torch.load(LOAD_PATH)

        enc.load_state_dict(state["enc"])
        enc_opt.load_state_dict(state["enc_opt"])
        dec.load_state_dict(state["dec"])
        dec_opt.load_state_dict(state["dec_opt"])
        EPOCH = state["epoch"]
        train_loss_history = state["train_loss_history"]

def show_loss():
    import matplotlib.pyplot as plt
    plt.plot(train_loss_history)
    plt.show()


if __name__ == "__main__":
    load_all()

    # Train
    crit = nn.MSELoss()

    while True:

        random.shuffle(DATA)

        tot_loss = 0
        batch_nr = 0

        start = time.time()
        print()
        for batch in range(0, len(DATA), BATCH_SIZE):
            print(
                "Epoch {} batch {} of {} ({:.2f}%): ".format(
                    EPOCH,
                    batch_nr,
                    len(DATA) // BATCH_SIZE,
                    (batch + BATCH_SIZE) / len(DATA) * 100
                ))

            if batch + BATCH_SIZE > len(DATA):
                sounds = [x.get_sound() for x in DATA[batch : ]];
            else:
                sounds = [x.get_sound() for x in DATA[batch : batch + BATCH_SIZE]]

            freqs = np.array([dataloader.slide_fft(x) for x in sounds])

            re_pos, re_neg, im_pos, im_neg = (
                np.real(freqs).reshape(freqs.shape[0], 1, *freqs.shape[1:]).clip(0, None),
                -np.real(freqs).reshape(freqs.shape[0], 1, *freqs.shape[1:]).clip(None, 0),
                np.imag(freqs).reshape(freqs.shape[0], 1, *freqs.shape[1:]).clip(0, None),
                -np.imag(freqs).reshape(freqs.shape[0], 1, *freqs.shape[1:]).clip(None, 0),
            )

            inp = np.log(np.concatenate([re_pos, re_neg, im_pos, im_neg], axis=1) + 1)

            print("\tEncoding...")
            encoded = enc(Variable(torch.Tensor(inp)))
            print("\tDecoding...")
            decoded = dec(encoded)

            loss = crit(decoded, Variable(torch.Tensor(inp)))
            print("\tLoss: {:.5f}. Backpropping...".format(loss.data[0]))

            enc_opt.zero_grad()
            dec_opt.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm(enc.parameters(), 10)
            nn.utils.clip_grad_norm(dec.parameters(), 10)

            print("\tStepping dec...")
            enc_opt.step()
            print("\tStepping enc...")
            dec_opt.step()

            if batch_nr % 10 == 0:
                print("\tSaving...")
                save()

            tot_loss += loss.data[0]

            batch_nr += 1
            time_per_batch = (time.time() - start) / batch_nr
            time_left = time_per_batch * (len(DATA) // BATCH_SIZE - batch_nr)

            print("\tDone")
            print(
                "\tTime left: {:02d}:{:02d}:{:02d}".format(
                    int(time_left // 3600),
                    int((time_left // 60) % 60),
                    int(time_left) % 60)
                )

        tot_loss /= (len(DATA) / BATCH_SIZE)

        print("Epoch loss: {}".format(tot_loss))
        train_loss_history.append(tot_loss)

        took = time.time() - start
        print(
            "Epoch {} took {:02d}:{:02d}:{:02d}".format(
                EPOCH,
                int(took // 3600),
                int((took // 60) % 60),
                int(took) % 60)
            )

        print("Saving")
        save()
        EPOCH += 1



    print("Done")

