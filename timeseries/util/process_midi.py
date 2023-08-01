""" For processing the ASAP Dataset. """

import os
import json

import numpy as np
import pandas
import music21

ASAP_PATH = "/stash/tlab/theom_intern/midi_data/asap-dataset-master"
METADATA = pandas.read_csv(os.path.join(ASAP_PATH, "metadata.csv"))
ASAP_ANNOTATIONS = json.load(open(os.path.join(ASAP_PATH, "asap_annotations.json")))


def parse_midi(path=None, id=None):
    if path:

        if not ASAP_ANNOTATIONS[path]["score_and_performance_aligned"]:
            print(f"Score & performance not aligned for {path}!")
            return None

        perf_path = os.path.join(ASAP_PATH, path)
        perf_ann_path = os.path.splitext(perf_path)[0]+'_annotations.txt'

        score_path = os.path.join(os.path.dirname(perf_path), "midi_score.mid")
        score_ann_path = os.path.splitext(score_path)[0]+'_annotations.txt'

        with open(perf_ann_path) as f:
            perf_ann = [[x[0], x[2]] for x in f.read().splitlines()]
        with open(score_ann_path) as f:
            score_ann = [[x[0], x[2]] for x in f.read().splitlines()]
        
        
        perf_stream = music21.converter.parse(perf_path, quantizePost=False).parts[0].flat
        score_stream = music21.converter.parse(score_path, quantizePost=False).flat

        perf_beats = ASAP_ANNOTATIONS[path]["performance_beats"]
        score_beats = ASAP_ANNOTATIONS[path]["midi_score_beats"]

        for e in perf_stream:
            if isinstance(e, music21.tempo.MetronomeMark):
                bps_perf = float(e.number)/60
                print(bps_perf)
                #break
        else:
            print(f"MISSING PERFORMANCE BPM {path}")
            #raise ZeroDivisionError

        bps_score = 0
        for e in score_stream:
            if isinstance(e, music21.tempo.MetronomeMark):
                if bps_score and bps_score != float(e.number)/60:
                    print(f"BPM CHANGES IN SCORE {path}")
                    raise ZeroDivisionError
                bps_score = float(e.number)/60
        if not bps_score:
            print(f"MISSING SCORE BPM {path}")
            raise ZeroDivisionError 
    

        shifted_notes = {}
        score_notes = {}

        for elem in perf_stream:
            if isinstance(elem, music21.chord.Chord):
                notes = [note for note in elem]
            elif isinstance(elem, music21.note.Note):
                notes = [elem]
            else:
                continue
            offset = elem.getOffsetBySite(perf_stream) / bps_perf

            for i, b_off in enumerate(perf_beats):
                if b_off > offset:
                    a = 0 if i == 0 else perf_beats[i-1]
                    b = b_off
                    beat_index = i
                    break
            else:
                a = perf_beats[-1]
                b = perf_beats[-1] + (perf_beats[-1]-perf_beats[-2])
                if offset > b:
                    b = offset + .01
                    print("LAST NOTE PAST LAST BEAT + 1!")
                    print(offset)
                    print(perf_beats[-2:])

            
            a_prime = 0 if beat_index == 0 else score_beats[beat_index-1]
            b_prime = score_beats[beat_index]

            new_offset = (b_prime - a_prime) / (b - a) * (offset - a) + a_prime

            for note in notes:
                if note.pitch.midi in shifted_notes:
                    shifted_notes[note.pitch.midi].append((new_offset, offset, note.volume.velocity))
                else:
                    shifted_notes[note.pitch.midi] = [(new_offset, offset, note.volume.velocity)]
        
        for elem in score_stream:
            if isinstance(elem, music21.chord.Chord):
                notes = [note for note in elem]
            elif isinstance(elem, music21.note.Note):
                notes = [elem]
            else:
                continue
            offset = elem.getOffsetBySite(score_stream) / bps_score

            for note in notes:
                if note.pitch.midi in score_notes:
                    score_notes[note.pitch.midi].append([offset, False, 0])
                else:
                    score_notes[note.pitch.midi] = [[offset, False, 0]]
        
        give_up = 0

        for pitch in shifted_notes:
            if pitch not in score_notes:
                #print(f"MISSING PITCH {pitch}: never played")
                continue
            score_offsets = np.asarray(score_notes[pitch])[:,0]

            for offset, old_offset, vel in shifted_notes[pitch]:
                idx_list = (np.abs(score_offsets - offset)).argsort()
                for idx in idx_list:
                    if abs(score_notes[pitch][idx][0] - offset) > 2:
                        give_up += 1
                        break
                    if score_notes[pitch][idx][1]:
                        if vel != score_notes[pitch][idx][2]:
                            continue
                        else:
                            break
                    else: 
                        score_notes[pitch][idx][1] = True
                        score_notes[pitch][idx][2] = vel
                        break
        
        missing = 0
        for i in score_notes:
            for j in score_notes[i]:
                if not j[1]:
                    missing += 1
        
        print(give_up, missing)

        return shifted_notes, score_notes







#parse_midi("Chopin/Scherzos/39/Bult-ItoS05M.mid")