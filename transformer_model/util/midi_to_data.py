import os
import numpy as np
import music21
from musicautobot.numpy_encode import NOTE_SIZE, SAMPLE_FREQ, MAX_NOTE_DUR, VALTCONT, VALTSEP, PIANO_RANGE, chordarr2stream
from musicautobot.utils import midifile
#x = "/stash/tlab/theom_intern/midi_data/little_pieces_4_(c)oguri.mid"

def stream2chordarr(s, note_size=NOTE_SIZE, sample_freq=SAMPLE_FREQ, max_note_dur=MAX_NOTE_DUR):
    "Converts music21.Stream to 1-hot numpy array"
    # assuming 4/4 time
    # note x instrument x pitch
    # FYI: midi middle C value=60
    
    # (AS) TODO: need to order by instruments most played and filter out percussion or include the channel
    highest_time = max(s.flat.getElementsByClass('Note').highestTime, s.flat.getElementsByClass('Chord').highestTime)
    maxTimeStep = round(highest_time * sample_freq)+1
    score_arr = np.zeros((maxTimeStep, len(s.parts), NOTE_SIZE))
    vol_arr = np.zeros((maxTimeStep, len(s.parts), NOTE_SIZE))

    def note_data(pitch, note, note2):
        return (pitch.midi, int(round(note.offset*sample_freq)), max(1, int(round(note.duration.quarterLength*sample_freq))), note2.volume.velocity)

    for idx,part in enumerate(s.parts):
        notes=[]
        for elem in part.flat:
            if isinstance(elem, music21.note.Note):
                notes.append(note_data(elem.pitch, elem, elem))
            if isinstance(elem, music21.chord.Chord):
                for note in elem.notes:
                    notes.append(note_data(note.pitch, elem, note))
                
        # sort notes by offset (1), duration (2) so that hits are not overwritten and longer notes have priority
        notes_sorted = sorted(notes, key=lambda x: (x[1], x[2])) 
        for n in notes_sorted:
            if n is None: continue
            pitch,offset,duration,velocity = n
            if max_note_dur is not None and duration > max_note_dur: duration = max_note_dur
            score_arr[offset, idx, pitch] = duration
            vol_arr[offset, idx, pitch] = velocity
            score_arr[offset+1:offset+duration, idx, pitch] = VALTCONT      # Continue holding note
    return score_arr, vol_arr

def chordarr2npenc(chordarr, volarr, skip_last_rest=True):
    # combine instruments
    result = []
    wait_count = 0
    last_vol = 0
    for idx,timestep in enumerate(chordarr):
        flat_time = timestep2npenc(timestep, volarr[idx])
        if len(flat_time) == 0:
            wait_count += 1
        else:
            # pitch, octave, duration, instrument
            if wait_count > 0: result.append([VALTSEP, wait_count, last_vol if last_vol > 0 else flat_time[0][2]])
            last_vol = flat_time[0][2]
            result.extend(flat_time)
            wait_count = 1
    #if wait_count > 0 and not skip_last_rest: result.append([VALTSEP, wait_count])
    return np.array(result, dtype=int).reshape(-1, 3) # reshaping. Just in case result is empty

# Note: not worrying about overlaps - as notes will still play. just look tied
# http://web.mit.edu/music21/doc/moduleReference/moduleStream.html#music21.stream.Stream.getOverlaps
def timestep2npenc(timestep, vol_timestep, note_range=PIANO_RANGE, enc_type=None):
    # inst x pitch
    notes = []
    for i,n in zip(*timestep.nonzero()):
        d = timestep[i,n]
        v = vol_timestep[i, n]
        difference = lambda input_list : abs(input_list - d)

        d_round = min([1,2,3,4,6,8,10,12,14,16,20,24,28,32], key=difference)
        if d < 0: continue # only supporting short duration encoding for now
        if n < note_range[0] or n >= note_range[1]:
            print("NOTE OUT OF RANGE: ", n)
            continue # must be within midi range
        notes.append([n,d_round,v,i])
        
    notes = sorted(notes, key=lambda x: x[0], reverse=True) # sort by note (highest to lowest)
    
    if enc_type is None: 
        # note, duration, vol
        return [n[:3] for n in notes]
    if enc_type == 'parts':
        raise NotImplementedError
    if enc_type == 'full':
        raise NotImplementedError

def parse_midi(path):
    mf = midifile.file2mf(path)
    stream = music21.midi.translate.midiFileToStream(mf, quarterLengthDivisors=(8,))
    chordarr, volarr = stream2chordarr(stream, sample_freq = 8)
    npenc = chordarr2npenc(chordarr, volarr)

    """for (note, length) in npenc:
        for j in current_lengths:
            current_lengths[j] += msg.time
        if msg.type != "note_on":
            continue
        print(msg)
        for j in current_lengths:
            current_lengths[j] += msg.time
        if msg.velocity > 0:
            notes.append([msg.note])
            vols.append(msg.velocity)
            most_recents[msg.note] = len(notes)-1
            current_lengths[msg.note] = 0
        else:
            notes[most_recents[msg.note]].append(current_lengths[msg.note])
            del current_lengths[msg.note]"""
    
    with open(os.path.splitext(path)[0]+'.txt', "w") as f:
        f.write('\n'.join('\t'.join(str(j) for j in i) for i in npenc))
    
def get_data(midi_path):
    text_path = os.path.splitext(midi_path)[0]+'.txt'
    if not os.path.isfile(text_path):
        print(f"First time loading: {midi_path}")
        parse_midi(midi_path)
    with open(text_path, 'r') as text:
        lines = text.readlines()
    data = [line.split('\t') for line in lines]
    notes = [list(map(int, d[0:2])) for d in data]
    vols = [int(d[2]) for d in data]

    return notes, vols