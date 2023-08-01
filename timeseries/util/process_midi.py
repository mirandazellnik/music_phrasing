""" For processing the ASAP Dataset. """

import os
import json

import numpy as np
import pandas
import mido

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
        
        
        perf_mf = mido.MidiFile(perf_path)
        score_mf = mido.MidiFile(score_path)

        perf_beats = ASAP_ANNOTATIONS[path]["performance_beats"]
        score_beats = ASAP_ANNOTATIONS[path]["midi_score_beats"]

        shifted_notes = {}
        score_notes = {}

        offset = 0

        for msg in perf_mf:
            if msg.type == "note_on" and msg.velocity > 0:
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

                
                if msg.note in shifted_notes:
                    shifted_notes[msg.note].append((new_offset, offset, msg.velocity))
                else:
                    shifted_notes[msg.note] = [(new_offset, offset, msg.velocity)]
            offset += msg.time
        
        offset = 0

        for msg in score_mf:
            if msg.type == "note_on" and msg.velocity > 0:
                if msg.note in score_notes:
                    score_notes[msg.note].append([offset, False, 0])
                else:
                    score_notes[msg.note] = [[offset, False, 0]]
            offset += msg.time
        
        give_up = 0

        for pitch in shifted_notes:
            if pitch not in score_notes:
                #print(f"MISSING PITCH {pitch}: never played")
                continue
            score_offsets = np.asarray(score_notes[pitch])[:,0]

            for offset, old_offset, vel in shifted_notes[pitch]:
                idx_list = (np.abs(score_offsets - offset)).argsort()
                for idx in idx_list:
                    if abs(score_notes[pitch][idx][0] - offset) > 1:
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