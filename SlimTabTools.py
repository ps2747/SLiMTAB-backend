import librosa
import numpy as np
import sounddevice as sd

note_name = ['C2', 'C2#', 'D2', 'D2#', 'E2', 'F2', 'F2#', 'G2', 'G2#', 'A2', 'A2#', 'B2',
              'C3', 'C3#', 'D3', 'D3#', 'E3', 'F3', 'F3#', 'G3', 'G3#', 'A3', 'A3#', 'B3', 
              'C4', 'C4#', 'D4', 'D4#', 'E4', 'F4', 'F4#', 'G4', 'G4#', 'A4', 'A4#', 'B4', 
              'C5', 'C5#', 'D5', 'D5#', 'E5', 'F5', 'F5#', 'G5', 'G5#', 'A5', 'A5#', 'B5',
              'C6', 'C6#', 'D6', 'D6#', 'E6', 'F6', 'F6#', 'G6', 'G6#', 'A6', 'A6#', 'B6',
              'C7', 'C7#', 'D7', 'D7#', 'E7', 'F7', 'F7#', 'G7', 'G7#', 'A7', 'A7#', 'B7',
              'C8', 'C8#', 'D8', 'D8#', 'E8', 'F8', 'F8#', 'G8', 'G8#', 'A8', 'A8#', 'B8']
openTab = [28, 23, 19, 14, 9, 4]
def TabRecgnize(position):
    return note_name[(openTab[position[0]] + position[1])%12]

def NoteDetection (audio, sr, thresh):
    cqt = np.average(librosa.core.cqt(y = audio, hop_length=2048,  sr = sr), axis = 1)
    note_contain = []
    p_note_contain = []
    for j in range(cqt.shape[0]):
        if cqt[j] >= cqt[max(0, j-1)] and cqt[j] >= cqt[min(cqt.shape[0]-1, j+1)]:
            if(cqt[j]>thresh) :
                note_contain.append(j)
                p_note_contain.append(note_name[j])
    #print(p_note_contain)
    return note_contain

##The tab input should be a size of 6 numpy array for six position
def TabCorrection(tabs, note_contain, open_tab =openTab):
    out_tabs = []
    press_num = 0
    #Check if there is only one tab is press
    for i in range(len(tabs)):
        if tabs[i] != 0:
            one_tab = [i+1, tabs[i]]
            press_num +=1
    #If there only one tab is pressed, then don't need to use audio aid
    if press_num == 1:
        return np.array(one_tab)
    #print('In tabs : ' + str(note_name[tabs[0]+ open_tab[0]]) + ' '+ str(note_name[tabs[1]+ open_tab[1]]) + ' '+ str(note_name[tabs[2]+ open_tab[2]]) + ' '+ str(note_name[tabs[3]+ open_tab[3]]) + ' '+ str(note_name[tabs[4]+ open_tab[4]]) + ' '+ str(note_name[tabs[5]+ open_tab[5]]) + ' ')
    for i in range(len(tabs)):
        note = tabs[i] + open_tab[i]
        closest_note = -1
        min_delta = 12
        if note_contain == None:
            return 0
        for i in reversed(range(len(note_contain))):
            delta = abs(note_contain[i] - note)

            #print('Tab in : ' +str(note_name[note]) + ' delta :' + str(delta))
            #if delta >= 2 we can't consider it as a candidate
            if delta < min_delta and delta < 3:
                closest_note = note_contain[i]
                min_delta = delta
                min_index = i
        #Only restore tabs that is attacked
        if closest_note - open_tab[i] > 0 and closest_note != -1:
            note_contain[min_index] = 100 #Make the used note to a very large number
            out_tabs = [i+1] + [closest_note - open_tab[i]] + out_tabs
    if out_tabs == []:
        out_tabs = [0]
    return np.array(out_tabs)

def valueSeparation(length, tab, min_value = 32):
    ret = []
    i = 0
    count = 0
    while length >= 1/32:
        num = int(length//(1/(2**i)))
        count += num
        for j in range(num):
            ret.append([2**i] + tab)
        length = length%(1/(2**i))
        i +=1
    if count > 1:
        for i in range(len(ret)-1):
            ret[i].append('c')
        ret[-1].append('e')
    return ret

if __name__ == '__main__':
    print(valueSeparation(0.375, [1, 0, 2, 4]))
