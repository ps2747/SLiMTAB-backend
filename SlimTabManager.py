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
    def __init__(self, samplerate=44100) :
        self.samplerate = samplerate
        self.bind_audio = np.array([])
        self.bind_tabdata = np.array([])

    def bindAudio(self, audiowave):
        self.bind_audio = audiowave
    
    def bindTabData(self, tabdata):
        self.bind_tabdata = tabdata

    #With corresponded audio data and tab data, use run to correct the tab data by using audio features
    def bindCalc(self, window_size = 2048, threshold = 0.8, samplerate = 44100):
        if self.bind_audio.size == 0 or self.bind_tabdata.size == 0:
            logging.warning('Binded data is(are) empty!!\n')
            return
        #Onset detection, label all the onsets and extract the note and tabs at that moment
        o_env = librosa.onset.onset_strength(self.bind_audio, sr = samplerate, aggregate = np.median, fmax = 8000, n_mels = 256)
        times = librosa.frames_to_time(np.arange(len(o_env)), sr = samplerate)
        
        onset_frames = librosa.onset.onset_detect(onset_envelope = o_env, sr= samplerate, backtrack = True)
        onset_samples = librosa.frames_to_samples(onset_frames)

        i = 0
        outputs = []
        for onset in onset_samples:
            note_contain = tls.NoteDetection(self.bind_audio[onset: onset + window_size], samplerate, threshold)
            onset_time = librosa.frames_to_time(onset, sr = samplerate)
            #Find the tabs where onset detected
            for j in range(i, bind_tabdata.size):
                if onset_time>= bind_tabdata[j][0] and onset_time < bind_tabdata[min(j+1, bind_tabdata.size)][0]:
                    tab_data =  bind_tabdata[j][1:]
                    i = j
                    break
            time_n_tabs = np.append(onset_time, tls.TabCorrection(tab_data, note_contain))
            outputs.append(time_n_tabs)
        outputs.append([self.bind_tabdata.size[-1][0], -1, -1, -1, -1, -1, -1])#set a pause note at the end
        ret = self._quantization(np.array(outputs))
        return ret

    def _quantization(self, data, bpm, time_sign_upper =4, time_sign_lower =4, min_note_value = 8, bypass_first_section = True):
        section_start = True
        quant_length = (60/bpm)*(time_sign_lower/min_note_value)
        section_time_length = 60/bpm*4*time_sign_upper/time_sign_lower
        outputs = []
        section_start_time = data[0][0]
        for i in range(data.shape[0]):
            if bypass_first_section:
                if data[i][0] < section_time_length:
                    continue
            if data[i][0] - section_start_time > section_time_length:
                section_start = True
            if section_start :
                section_start = False
                sum_note_value = 0
                section_start_time += section_time_length
                #For some situation that note don't start at the first beat of section
                if data[i][0] -section_start_time >= quant_length:
                    pause_note_length = data[i][0] -section_start_time
                    if pause_note_length%quant_length <= quant_length/2:
                        pause_note_valuetime = qunat_length*pause_note_length//quant_length 
                    else:
                        pause_note_valuetime = qunat_length*(pause_note_length//quant_length +1)
                    pause_note_value = 60/bpm/pause_note_valuetime*time_sign_lower
                    outputs.append([pause_note_value, -1, -1, -1, -1, -1, -1])
                    sum_note_value += 1/(60/bpm/pause_note_valuetime*time_sign_lower)
            note_length = data[min(data.shape[0]-1, i+1)][0] - data[i][0]
            if note_length == 0:
                continue
            if note_length%quant_length <= quant_length/2:
                note_valuetime = quant_length*(note_length//quant_length)
            else:
                note_valuetime = quant_length*(note_length//quant_length +1)
            note_value = 60/bpm/note_valuetime * time_sign_lower
            
            outputs.append([note_value, int(data[i][1]), int(data[i][2]), int(data[i][3]), int(data[i][4]), int(data[i][5]), int(data[i][6])])
            sum_note_value += 1/note_value
            if sum_note_value >= 1:
                section_start = True
        return np.array(outputs)                


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
        if curr_devices[self.device['index']]['name'] == self.device['name']:
            return True
        else:
            return False
    
    def calc(self):
        ad = AudioAid()
        ad.bindAudio(self.record_ardata)
        ad.bindTabData(self.record_trdata)
        merge_data = ad.bindCalc()


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
        
    def saveCurrentRecordData(self, name = None):
        import tempfile
        if name is None or name == '' :
            name = tempfile.mktemp(prefix='rec_',suffix='.wav', dir='')
        with sf.SoundFile(name, mode='x', samplerate=self.samplerate, channels = self.device['max_input_channels']) as file:
            file.write(np.reshape(self.record_ardata, (-1, 2)))
        return name

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
            tab_driver = driver.SliMTABDriver("/dev/tty.SLAB_USBtoUART")
        except Exception as exception:
            logging.warning('\nFail to access Tab driver data: ' + str(exception))
            self.b.wait()
            return
        record_tabs = []
        self.b.wait()
        self.driver_check = tab_driver.check()
        if not tab_driver.open() or not self.driver_check:
            logging.warning('Tab device open unsucceed')
            return 
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
        return stream

    def _getInputDevices(self):
        devices = sd.query_devices()
        
        default_input_device = sd.query_devices(kind = 'input')

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
                test_data = np.array([[0.0122, 1, 0, 0, 0, 0, 0], [1.03322, 1, 0, 0, 0, 0, 0], [1.543423, 2, 0, 0, 0, 0, 0], [1.7523423, 3, 0, 0, 0, 0, 0], [2.15435, -1, -1, -1, -1, -1, -1], [2.362345, -1, -1, -1, -1, -1, -1]])
                print(aa.quantization(test_data, 120))
            else:
                print('Invalid input!!')    
