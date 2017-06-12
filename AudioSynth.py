#!/usr/bin/python3
import numpy as np
import time

class AudioSynth:
    def __init__(self, sr=44100):
        self.sr = sr
        self.data = np.zeros(44100)
        self.start_time = 0
        self.noise_pool = {}

    def gen_signal(self, start, N, freq, phase=0):
        n = int(self.sr/freq*2)
        noise = np.random.rand(n)-0.5
        theta0 = int(phase*n)
        for i in range(N):
            noise[(i+n-theta0)%n] = 0.5*(noise[(i+n+1-theta0)%n]+noise[(i+n-theta0)%n])
            self.data[i+start] += noise[(i+n-theta0)%n]

    def gen(self, l):
        self.data = np.zeros(int((l[-1]['time']+l[-1]['duration'])*self.sr)+100)
        for nt in l:
            freq = 82.41*np.power(1.0594630943592953, nt['note']-28)
            self.gen_signal(int(nt['time']*self.sr), int(nt['duration']*self.sr), int(freq), np.random.random())

    def getCurrentTime(self):
        return time.perf_counter() - self.start_time

    def play(self):
        import sounddevice as sd
        self.start_time = time.perf_counter()
        sd.play(self.data)
