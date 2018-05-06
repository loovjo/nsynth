import os
import random
import sounddevice as sd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

import dataloader

_, SAMPLE_RATE, SAMPLE_LENGTH = dataloader.load_data(amount=1, silent=True)

CTX_SIZE = 500
SAMPLE_GEN = 400
BATCH_SIZE = 50

SAVE_PATH = LOAD_PATH = "ai.pt"

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv_1 = nn.Conv1d(1, 100, 50)
        self.conv_2 = nn.Conv1d(100, 50, 50)
        self.conv_3 = nn.Conv1d(50, 10, 50)
        self.conn   = nn.Linear(4990, CTX_SIZE)

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)

        x = self.conv_1(x)
        x = nn.MaxPool1d(5)(x)
        x = self.conv_2(x)
        x = nn.MaxPool1d(5)(x)
        x = self.conv_3(x)
        x = nn.MaxPool1d(5)(x)

        x = x.view(x.shape[0], -1)
        x = self.conn(x)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.last_proc = nn.Linear(SAMPLE_GEN, 100)
        self.ctx_proc  = nn.Linear(CTX_SIZE, 100)
        self.proc_both = nn.Linear(200, 500)
        self.proc_out  = nn.Linear(500, SAMPLE_GEN)
        self.proc_ctx  = nn.Linear(500, CTX_SIZE)

    def forward(self, last, ctx):
        last = self.last_proc(last)
        ctx = self.ctx_proc(ctx)

        combined = torch.cat((last, ctx), dim=1)
        proc = self.proc_both(combined)
        out = self.proc_out(proc)
        new_ctx = self.proc_ctx(proc)

        return out, new_ctx


def test_show(amount=10):
    sounds = [x.sound for x in data[:amount]]
    print(data[0])
    sound = Variable(torch.Tensor(sounds))
    print(sound)
    ctx = enc(sound)
    print(ctx)

    result = np.zeros((len(sounds), 1))
    for i in range(0, SAMPLE_LENGTH, SAMPLE_GEN):
        if i + SAMPLE_GEN > len(sound[0]):
            break

        print(i, ":", i + SAMPLE_GEN)
        last_out = sound[:, i:i+SAMPLE_GEN].contiguous()
        out, ctx = dec(last_out, ctx)
        result = np.concatenate((result, out.data.numpy()), axis=1)

    print("Done")
    plt.plot(result.T)
    plt.show()

    for i, x in enumerate(result):
        plt.plot(sounds[i])
        plt.plot(x)
        plt.show()
        sd.play(sounds[i], blocking=True)
        sd.play(x, blocking=True)

def save():
    save_dict = {
            "enc": enc,
            "enc_opt": enc_opt,
            "dec": dec,
            "dec_opt": dec_opt,
            "epoch": EPOCH,
            "train_loss_history": train_loss_history,
            }

    torch.save(save_dict, SAVE_PATH)

def load_all():
    global enc, dec, enc_opt, dec_opt, EPOCH, train_loss_history, data

    enc = Encoder()
    dec = Decoder()

    enc_opt = optim.SGD(enc.parameters(), lr=0.1, momentum=0.1)
    dec_opt = optim.SGD(dec.parameters(), lr=0.1, momentum=0.1)
    EPOCH = 0
    train_loss_history = []


    print("Load old state? [Y/n] ", flush=True, end="")

    if os.path.isfile(LOAD_PATH) and input().lower() == "y":
        print("Loading old state")
        state = torch.load(LOAD_PATH)

        enc = state["enc"]
        enc_opt = state["enc_opt"]
        dec = state["dec"]
        dec_opt = state["dec_opt"]
        EPOCH = state["epoch"]
        train_loss_history = state["train_loss_history"]

    data, _, _ = dataloader.load_data()

if __name__ == "__main__":
    load_all()


    # Train
    crit = nn.MSELoss()

    while True:

        print("Epoch", EPOCH, "training")
        enc_opt.zero_grad()
        dec_opt.zero_grad()

        random.shuffle(data)
        sounds = [x.sound for x in data[:BATCH_SIZE]]
        sound = Variable(torch.Tensor(sounds))

        loss = Variable(torch.zeros(1))

        ctx = enc(sound)
        print("encoded")

        for i in range(0, SAMPLE_LENGTH, SAMPLE_GEN):
            print(i - SAMPLE_GEN, ":", i, ":", i + SAMPLE_GEN)
            if i + SAMPLE_GEN > len(sound[0]):
                break

            if i >= SAMPLE_GEN:
                last_out = sound[:, i-SAMPLE_GEN:i].contiguous()
            else:
                last_out = Variable(torch.zeros(len(sound), SAMPLE_GEN))

            wanted = sound[:, i:i+SAMPLE_GEN]

            got, ctx = dec(last_out, ctx)

            batch_loss = crit(got, wanted)
            loss += batch_loss

        loss /= (SAMPLE_LENGTH / SAMPLE_GEN)

        print("Loss =", loss.data[0])

        print("Backward")
        loss.backward()
        print("Step enc")
        enc_opt.step()
        print("Step dec")
        dec_opt.step()

        print("Saving")
        train_loss_history.append(loss.data[0])

        EPOCH += 1

        save()


    print("Done")


    test_show()
