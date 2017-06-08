import librosa
import numpy as np
import sounddevice as sd

note_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
openTab = [4, 11, 7, 2, 9, 4]
def TabRecgnize(position):
    return note_name[(openTab[position[0]] + position[1])%12]

def NoteDetection (audio, sr, thresh):
    chroma_cqt = np.average(librosa.feature.chroma_cqt(y = audio, hop_length=2048,  sr = sr), axis = 1)
    note_contain = []
    for j in range(chroma_cqt.shape[0]):
        if chroma_cqt[j] >= chroma_cqt[max(0, j-1)] and chroma_cqt[j] >= chroma_cqt[min(chroma_cqt.shape[0]-1, j+1)]:
            if(chroma_cqt[j]>thresh) :
                note_contain.append(note_name[j]+str(chroma_cqt[j]))
    return note_contain

##The tab input should be a size of 6 numpy array for six position
def TabCorrection(tabs, note_contain, open_tab =openTab):
    out_tabs = []
    for i in range(len(tabs)):
        note = (tabs[i] + open_tab[i])%12
        closest_note = -1
        min_delta = 12
        for note_candi in note_contain:
            delta = abs(note_candi - note)
            #if delta >= 2 we can't consider it as a candidate
            if delta < min_delta and delta < 2:
                closest_note = note_candi
                min_delta = delta
        #Only restore tabs that is attacked
        if closest_note != -1:
            out_tabs.append([i, closest_note - open_tab[i]])
    return out_tabs

        
