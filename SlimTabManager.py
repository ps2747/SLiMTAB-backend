import numpy as np
import sys

import librosa
import threading
import time
import queue

import tempfile
import logging

import sounddevice as sd
import soundfile as sf

import SlimTabTools as tls
import SlimTabDriver as driver

class AudioAid:
    #Notice that audio recorder and tab driver have different sample rate, they should compute in different time domain
    def __init__(self, samplerate=44100, bpm = 120, sign_upper = 4, sign_lower = 4, min_note_value = 8, bypass_first_bar = True) :
        self.samplerate = samplerate
        self.bpm = bpm
        self.sign_upper = sign_upper
        self.sign_lower = sign_lower
        self.min_note_value = min_note_value
        self.bypass_first_bar = bypass_first_bar
        self.bind_audio = np.array([])
        self.bind_tabdata = np.array([])

    def bindAudio(self, audiowave):
        self.bind_audio = audiowave
    
    def bindTabData(self, tabdata):
        self.bind_tabdata = tabdata
    
    def setArgs(self, bpm = 120, sign_upper = 4, sign_lower = 4, min_note_value = 8, bypass_first_bar = True):
        self.bpm = bpm
        self.sign_upper = sign_upper
        self.sign_lower = sign_lower
        self.min_note_value = min_note_value
        self.bypass_first_bar = bypass_first_bar

    #With corresponded audio data and tab data, use run to correct the tab data by using audio features
    def calcResult(self, window_size = 2048, threshold = 1.0e-2, samplerate = 44100):
        if self.bind_audio.size == 0 : 
            logging.warning('Binded audio data is(are) empty!!\n')
            return
        if self.bind_tabdata.size == 0:
            logging.warning('Binded tab data is(are) empty!!\n')
            return
        #Onset detection, label all the onsets and extract the note and tabs at that moment
        mono = librosa.core.to_mono(self.bind_audio.T)
        #mono = self.bind_audio[2][:]
        o_env = librosa.onset.onset_strength(mono, sr = samplerate, aggregate = np.median, fmax = 8000, n_mels = 256)
        times = librosa.frames_to_time(np.arange(len(o_env)), sr = samplerate)
        
        onset_frames = librosa.onset.onset_detect(onset_envelope = o_env, sr= samplerate, backtrack = True)
        onset_samples = librosa.frames_to_samples(onset_frames)

        i = 0
        outputs = []
        for onset in onset_samples:
            note_contain = tls.NoteDetection(mono[onset: onset + window_size], samplerate, threshold)
            #print(note_contain)
            onset_time = onset/samplerate *1000
            #Find the tabs where onset detected
            for j in range(i, self.bind_tabdata.shape[0]):
                if onset_time < self.bind_tabdata[0][0]:
                    tab_data = self.bind_tabdata[0][1:]
                    i = 0
                    break
                if onset_time>= self.bind_tabdata[j][0] and onset_time < self.bind_tabdata[min(j+1, self.bind_tabdata.shape[0]-1)][0]:
                    tab_data =  self.bind_tabdata[j][1:]
                    i = j
                    break
            tabs = tls.TabCorrection(tab_data, note_contain)
            time_n_tabs = [onset_time] + tabs.tolist()
            outputs.append(time_n_tabs)
        outputs.append([self.bind_tabdata[-1][0]])#set a pause note at the end
        ret = self._quantization(np.array(outputs))
        return ret    

    def _quantization(self, data):
        quant_length = (60/self.bpm)*(self.sign_lower/self.min_note_value)
        bar_length = self.sign_upper/self.sign_lower
        bar_start_time = 0

        #Quantize and remap data
        for i in range(data.shape[0]):
            if data[i] is None:
                continue
            if data[i][0]%quant_length <= quant_length/2:
                data[i][0]=(data[i][0]/1000.0//quant_length)*(1/self.min_note_value)
            else:
                data[i][0]=(data[i][0]/1000.0//quant_length + 1)*(1/self.min_note_value)

        #If bypass first bar is True, delete all note which note time below 1
        if self.bypass_first_bar:
            bar_start_time = 1
            i = 0
            while i < data.shape[0] and  data[i][0] <= 1:
                data = np.delete(data, i, 0)
                i+= 1
        #To fill the gap with pause between start time and the first data 
        if data[0][0] > bar_start_time:
            data = np.array([[bar_start_time, 0]] + data.tolist())
        
        for i in range(data.shape[0]):
            data[i][0] = data[min(data.shape[0]-1, i+1)][0] - data[i][0]
        
        #Delet item that has the same time as the latter item  
        while i < data.shape[0]:
            if data[i][0] == 0:
                data = np.delete(data, i, 0)
            else:
                i += 1
        
        bars = []
        bar = []
        sum_len = 0
        #Map data to sheet music template
        for note in data:
            note_len = note[0]
            while note_len > 0:
                if sum_len + note_len >= 1:
                    fill_note = 1 - sum_len
                    separated_note = tls.valueSeparation(fill_note, note[1:])
                    bar = bar + separated_note
                    #bar.append([fill_note] + note[1:])
                    bars.append(bar)
                    bar = []
                    note_len -= fill_note
                    sum_len = 0
                else:
                    separated_note = tls.valueSeparation(note_len, note[1:])
                    bar = bar + separated_note
                    #bar.append([note_len] + note[1:])
                    sum_len += note_len
                    note_len = 0
        
        if bar != []:
            bars.append(bar)
        return bars

class SlimTabManager:
    def __init__(self,) :
        self.record_audios = np.array([])
        self.record_names = {}#Dictionary
        self.record_tab_datas = np.array([])
        self.record_ardata = np.array([])
        self.record_trdata = np.array([])
        self.temp_array = []

        self.audio_aid = AudioAid()
        self.stop_key = False
        self.record_status = 0

        self.q = queue.Queue()
        self.b = threading.Barrier(2, timeout = 5)
        self.sync_stop_key = False
        self.input_devices = self._getInputDevices()
        self.i = 0

        #For UI audio record wave randering
        self.this_wavelet=np.array([])

        for device in self.input_devices:
            try:
                self.input_stream = self._openRecordStream(device, device['default_samplerate'])
                self.device = device
                self.samplerate = int(device['default_samplerate'])

                break
            except Exception as exception:
                self.device = None
                logging.warning('Fail to open stream: ' + str(exception))

    def check(self):
        curr_devices = self._getInputDevices()
        status = False
        for device in curr_devices:
            if self.device != None and device['name'] == self.device['name']:
                status = True
        return status
    
    def calc(self, ar_data = None, tr_data = None, bpm = 120, sign_upper = 4, sign_lower = 4, min_note_value = 8, bypass_first_bar = True):
        if ar_data == None:
            ar_data = self.record_ardata
        if tr_data == None:
            tr_data = self.record_trdata
        self.audio_aid.bindAudio(ar_data)
        self.audio_aid.bindTabData(tr_data)
        self.audio_aid.setArgs(bpm, sign_upper, sign_lower, min_note_value, bypass_first_bar)
        return self.audio_aid.calcResult()


    def record(self, filename = ''):
        if self.record_status != 0:
            print("Invalid action, record have been started.")
            return
        self.stop_key = False
        self.sync_stop_key = False
        self.tTR = threading.Thread(target = self._tTabRecord)
        self.tRC = threading.Thread(target = self._tRecordConsume)
        self.start_time = time.time()
        self.tTR.start()
        if self.check() :
            self.record_status = 1
            if self.input_stream.active:
                self.input_stream.stop()
            self.b.wait()
            self.input_stream.start()
            self.tRC.start()
        else:
            self.record_status = 0
        
    def stopRecord(self):
        if self.record_status == 0:
            return 
        self.record_status = 0
        self.stop_time = time.time()
        if self.check():
            self.input_stream.stop()
            self.record_ardata = np.reshape(np.array(self.temp_array), (-1, 2))
        self.stop_key = True
        try: 
            self.tTR.join()
            self.tRC.join()      
        except Exception:
            return
        self.record_ardata= self._monoToStereo(self.record_ardata)
        
    def saveCurrentRecordData(self, name = None):
        import tempfile
        if name is None or name == '' :
            name = tempfile.mktemp(prefix='rec_',suffix='', dir='')
            tab_name = name +'.tab'
        with sf.SoundFile(name + '.wav', mode='x', samplerate=self.samplerate, channels = self.device['max_input_channels']) as file:
            file.write(np.reshape(self.record_ardata, (-1, 2)))
        tab_file = open(tab_name, 'w')
        for tab in self.record_trdata:
            tab_file.write('%s\n' % tab)

        return name
    
    def loadRecordData(self, name = None):
        try:
            self.record_ardata, sr = librosa.load(name + '.wav', sr = self.samplerate)
        except Exception:
            print('Fail to load audio data')
            return 
        try:
            with open(name + '.tab') as file:
                tabs = []
                for line in file:
                    line = line[1:-2].split(' ')
                    line = [int(l) for l in line if l != '']
                    tabs.append(line)
                self.record_trdata = np.array(tabs)
        except Exception:
            print('Fail to load tab data')
            return 

    def close(self):
        self.record_status = -1
        self.stopRecord()
        self.input_stream = None
        self.device = None
    
    def setInputDevice(self, deviceIndex):
        try:
            self.device = self.input_devices[deviceIndex]
            self.input_stream = self._openRecordStream(self.device, self.samplerate)
            self.samplerate = int(self.device['default_samplerate'])
        except Exception as exception:
            self.device = None
            logging.warning('Fail to open stream: ' + str(exception))

    def getInputDevicesName(self):
        outputs = []
        self._getInputDevices()
        for device in self.input_devices:
            outputs.append(device['name'])
        return outputs

    def getCurrDeviceName(self):
        return self.device['name']

    def getWaveletUpdataFreq(self):
        return self.samplerate/512
    
    def getWavelet(self):
        return self.this_wavelet

    def getAudioRecorderInfo(self):
        self.samplerate = samplerate
        return self.samplerate, self.device

    def getAudioWave(self, name = None):
        if name is None:
            return self.record_ardata
        else:
            return self.record_audios[self.record_names[name]]

    def getDefaultDevice(self):
        return {'input':sd.default.device['input'], 'output': sd.default.device['output']} 

    def printTime(self):
        print('Record time :' + str(len(self.temp_array)*len(self.this_wavelet)/self.samplerate))
        if self.driver_check:
            print('Driver record time:' + str(self.tabRT))
        print('Now time :' + str(time.time() - self.start_time))

    def print(self):
        print(self.record_ardata.shape)
        print(self.record_trdata.shape)
        print('Tab record time : ' + str(self.record_trdata[-1][0]/1000))
        print('Audio record time : ' + str(self.record_ardata.shape[0]/self.samplerate)) 
        print('Delta : ' + str(self.record_trdata[-1][0]/1000 - self.record_ardata.shape[0]/self.samplerate))
        print('Timer : ' + str(self.stop_time - self.start_time))

    def _callback(self, indata, frames, time, status):
        if status:
            logging.error(status, file = sys.stderr)
        self.q.put(indata.copy())

    def _tTabRecord(self):
        try:
            tab_driver = driver.SliMTABDriver("192.168.100.1")
        except Exception as exception:
            logging.warning('\nFail to access Tab driver data: ' + str(exception))
            tab_driver.close()
            self.b.wait()
            return
        record_tabs = []
        self.b.wait()
        if tab_driver.check() == -1:
            logging.warning('Tab device open unsucceed')
            return 
        tab_driver.open()
        tab_driver.reset()
        tab_driver.begin()
        i = 0 
        while not self.stop_key:
            i += 1
            ts, n, tab = tab_driver.read()
            if self.stop_key :
                break
            record_tabs.append([ts] + [t for t in tab])
            self.tabRT = ts
        self.record_trdata = np.array(record_tabs)
        tab_driver.end()
        tab_driver.close()

    def _tRecordConsume(self):
        i = 0
        while not self.stop_key:
            if not self.q.empty():
                self.this_wavelet = self.q.get()
                self.temp_array.append(self.this_wavelet.flatten())
        while not self.q.empty():
            self.this_wavelet = self.q.get()
            self.temp_array.append(self.this_wavelet.flatten())
        #print('Consumes end')
    
    def _openRecordStream(self, device, sr = 44100, ):
        logging.info('Openning the stream for device: '+str(device['name']))
        stream = sd.InputStream(samplerate = sr, blocksize = 4096, device = device['index'], channels = device['max_input_channels'], callback = self._callback)
        #stream = sd.InputStream(samplerate = sr, blocksize = 4096, device = device['index'], channels = 1, callback = self._callback)
        return stream

    def _getInputDevices(self):
        devices = sd.query_devices()
        
        if sd.default.device['input'] == -1:
            default_input_device = None
        else:
            default_input_device = devices[sd.default.device['input']]

        input_devices = []
        if default_input_device is not None:
            default_input_device['index'] = devices.index(default_input_device)
            input_devices += [default_input_device]
        
        for device in devices:
            if device['max_input_channels']> 0:
                device['index'] = devices.index(device)

                #In windows, there might be no defaults
                if default_input_device is None or device['index'] != default_input_device['index']:
                    input_devices += [device]
        return input_devices
    
    def _monoToStereo(self, audio):
        for i in range(audio.shape[0]):
            audio[i][0]=audio[i][1]
        return audio


if __name__ == '__main__':
    import time
    import sys
    import select
    def isData():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
    
    terminate_key = False
    manager = SlimTabManager()
    while not terminate_key:
        if isData():
            line = sys.stdin.readline() 
            cmd, arg = (line[:-1]+' ').split(' ', 1)
            if cmd == 'start':
                manager.record()

            elif cmd == 'stop':
                manager.stopRecord()
                rlt = manager.calc()
                print(rlt)
                #for sec in rlt:
                #    line = []
                #    for r in  sec:
                #        line.append(r)
                    #print(line)
            elif cmd == 'calc':
                rlt = manager.calc()
                print(rlt)

            elif cmd == 'save_current':
                filename = manager.saveCurrentRecordData(name = arg[:-1])
                logging.info('Save file name : ' + filename)

            elif cmd == 'get_wavelet':
                print(manager.getWavelet())
            
            elif cmd == 'get_wavelet_sr':
                print(manager.getWaveletUpdataFreq())

            elif cmd == 'get_devices':
                print(manager.getInputDevicesName())
            elif cmd == 'set_device':
                manager.setInputDevice(int(arg))
            
            elif cmd == 'exit':
                manager.close()
                break
            elif cmd == 'print':
                manager.printTime()
            elif cmd == 'quant':
                aa = AudioAid()
                test_data = np.array([[1033.22, 1, 1], [4324.234234, 1, 2], [5234.1234, 1, 3], [6234.1341234, 1, 4], [6734.125135, 1, 5], [7123.41234124, 1, 6]])
                #test_data = np.array([[0.0122, 1, 0, 0, 0, 0, 0], [1.03322, 1, 0, 0, 0, 0, 0], [1.543423, 2, 0, 0, 0, 0, 0], [1.7523423, 3, 0, 0, 0, 0, 0], [2.15435, -1, -1, -1, -1, -1, -1], [2.362345, -1, -1, -1, -1, -1, -1]])
                if len(arg) == 0:
                    arg = 120
                bpm = int(arg) 
                print('bpm : ' + str(bpm))
                aa.setArgs(bpm = bpm)
                print(aa._quantization(test_data))
            elif cmd == 'default_name':
                print(manager.getDefaultDeviceName())
            elif cmd == 'check':
                manager.check()
            elif cmd == 'load':
                manager.loadRecordData('rec_hpv88qn3')
            else:
                print('Invalid input!!')    
