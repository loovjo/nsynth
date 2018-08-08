import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

import dataloader

_, SAMPLE_RATE, SAMPLE_LENGTH = dataloader.load_data(amount=1, silent=True)

CTX_SIZE = 500
SAMPLE_GEN = 400

BATCH_SIZE = 20

TEACHER_FORCE_RATE = 0.1

SAVE_PATH = LOAD_PATH = "ai.pt"

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv_1 = nn.Conv1d(1, 50, 50)
        self.conv_2 = nn.Conv1d(50, 10, 50)
        self.conn   = nn.Linear(6340, CTX_SIZE)

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)

        x = self.conv_1(x)
        x = nn.MaxPool1d(10)(x)
        x = self.conv_2(x)
        x = nn.MaxPool1d(10)(x)

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


def test_show(amount=10, force_teacher=True):
    import sounddevice as sd
    import matplotlib.pyplot as plt


    sounds = [x.get_sound() for x in data[:amount]]
    print(data[0])
    sound = Variable(torch.Tensor(sounds))
    # Normalize
    sound /= sound.data.std()
    print(sound.data.std())

    print(sound)
    ctx = enc(sound)
    print(ctx)

    if force_teacher:
        result = np.zeros((len(sounds), 1))
    else:
        result = sound[:,:SAMPLE_GEN].data.numpy()
    for i in range(0, SAMPLE_LENGTH, SAMPLE_GEN):
        if i + SAMPLE_GEN > len(sound[0]):
            break

        print(i, ":", i + SAMPLE_GEN)
        if force_teacher:
            last_out = sound[:, i:i+SAMPLE_GEN].contiguous()
        else:
            last_out = Variable(torch.Tensor(result[:,i:i+SAMPLE_GEN]))
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
            "enc": enc.state_dict(),
            "enc_opt": enc_opt.state_dict(),
            "dec": dec.state_dict(),
            "dec_opt": dec_opt.state_dict(),
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

    if os.path.isfile(LOAD_PATH) and input().lower() != "n":
        print("Loading old state")
        state = torch.load(LOAD_PATH)

        enc.load_state_dict(state["enc"])
        enc_opt.load_state_dict(state["enc_opt"])
        dec.load_state_dict(state["dec"])
        dec_opt.load_state_dict(state["dec_opt"])
        EPOCH = state["epoch"]
        train_loss_history = state["train_loss_history"]

    data, _, _ = dataloader.load_data()


def show_loss():
    import matplotlib.pyplot as plt
    plt.plot(train_loss_history)
    plt.show()


if __name__ == "__main__":
    load_all()


    # Train
    crit = nn.MSELoss()

    while True:

        print("Epoch", EPOCH, "training")
        enc_opt.zero_grad()
        dec_opt.zero_grad()

        random.shuffle(data)

        tot_loss = 0
        batch_nr = 0

        start = time.time()
        iter = 0
        print()
        for batch in range(0, len(data), BATCH_SIZE):
            print("\033[1A\033[KBatch {} of {}: ".format(batch_nr, len(data) // BATCH_SIZE), end="", flush=True)
            if batch + BATCH_SIZE > len(data):
                sounds = [x.get_sound() for x in data[batch : ]];
            else:
                sounds = [x.get_sound() for x in data[batch : batch + BATCH_SIZE]]

            sound = Variable(torch.Tensor(sounds))
            # Normalize
            sound /= sound.data.std()

            loss = Variable(torch.zeros(1))

            last_out = None

            print("encoding...", end="", flush=True);
            ctx = enc(sound)
            print(" - DONE. Generating...", end="", flush=True)

            for i in range(0, SAMPLE_LENGTH, SAMPLE_GEN):
                if i + SAMPLE_GEN > len(sound[0]):
                    break

                if random.random() < TEACHER_FORCE_RATE or last_out is None:
                    if i >= SAMPLE_GEN:
                        last_out = sound[:, i-SAMPLE_GEN:i].contiguous()
                    else:
                        last_out = Variable(torch.zeros(len(sound), SAMPLE_GEN))

                wanted = sound[:, i:i+SAMPLE_GEN]

                got, ctx = dec(last_out, ctx)

                batch_loss = crit(got, wanted)
                loss += batch_loss

                last_out = got

            loss /= (SAMPLE_LENGTH / SAMPLE_GEN)

            print(" - DONE. Loss: {:.5f}. Backpropping...".format(loss.data[0]), end="", flush=True)

            loss.backward()
            print(" - DONE. Stepping enc...", end="", flush=True)
            enc_opt.step()
            print(" - DONE. Stepping dec...", end="", flush=True)
            dec_opt.step()

            if iter % 20 == 0:
                print(" - DONE. Saving...", end="", flush=True)
                save()

            iter += 1

            print(" - DONE.")

            tot_loss += loss.data[0]
            batch_nr += 1

        tot_loss /= (len(data) / BATCH_SIZE)

        print("Epoch loss: {}".format(tot_loss))
        train_loss_history.append(tot_loss)

        print("Took {}s", time.time() - start)

        print("Saving")
        save()
        EPOCH += 1



    print("Done")

