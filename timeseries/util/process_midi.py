""" For processing the ASAP Dataset. """

import os
import json

import numpy as np
import pandas
import mido

ASAP_PATH = "/stash/tlab/theom_intern/midi_data/asap-dataset-master"
PROCESSED_PATH = "/stash/tlab/theom_intern/midi_data/asap-dataset-processed"

METADATA = pandas.read_csv(os.path.join(ASAP_PATH, "metadata.csv"))
ASAP_ANNOTATIONS = json.load(open(os.path.join(ASAP_PATH, "asap_annotations.json")))


def parse_midi(path=None, id=None):
    if path:

        if not ASAP_ANNOTATIONS[path]["score_and_performance_aligned"]:
            print(f"Score & performance not aligned for {path}! Skipping.")
            return None, None, None

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

        all_timed_score_vels = []
        all_timed_score_lengths = []

        offset = 0
        playing_note_times = {}

        for msg in perf_mf:
            if msg.type == "note_on":
                if msg.velocity > 0:
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
                            print(f"Note past last bar in {path}!")

                    
                    a_prime = 0 if beat_index == 0 else score_beats[beat_index-1]
                    b_prime = score_beats[beat_index]

                    new_offset = (b_prime - a_prime) / (b - a) * (offset - a) + a_prime
                    playing_note_times[msg.note] = 0
                    
                    if msg.note in shifted_notes:
                        shifted_notes[msg.note].append([new_offset, False, msg.velocity, 0])
                    else:
                        shifted_notes[msg.note] = [[new_offset, False, msg.velocity, 0]]
                else:
                    shifted_notes[msg.note][-1][3] = playing_note_times[msg.note]
                    del playing_note_times[msg.note]

            offset += msg.time
            for key in playing_note_times:
                playing_note_times[key] += msg.time
        
        offset = 0
        playing_note_times = {}
        
        for msg in score_mf:
            offset += msg.time
            for key in playing_note_times:
                playing_note_times[key] += msg.time
            
            if msg.type == "note_on":

                if msg.velocity > 0:
                    if msg.note in playing_note_times:
                        continue # DOUBLE NOTE???
                    if msg.note in score_notes:
                        if score_notes[msg.note][-1][0] == offset: # If there is a 0-length at this location already, reset it
                            score_notes[msg.note][-1] = [offset, False, 0, len(all_timed_score_vels), 0, -1, True]
                            playing_note_times[msg.note] = 0
                            continue
                        score_notes[msg.note].append([offset, False, 0, len(all_timed_score_vels), 0, -1, True])
                    else:
                        score_notes[msg.note] = [[offset, False, 0, len(all_timed_score_vels), 0, -1, True]]
                    all_timed_score_vels.append(0)
                    all_timed_score_lengths.append(0)

                    playing_note_times[msg.note] = 0
                else:

                    if msg.note not in playing_note_times:
                        continue # DOUBLE END_NOTE?
                    score_notes[msg.note][-1][5] = playing_note_times[msg.note]

                    del playing_note_times[msg.note]


            
        
        give_up = 0

        for pitch in score_notes:
            if pitch not in shifted_notes:
                #print(f"MISSING PITCH {pitch}: never played")
                continue
            shifted_offsets = np.asarray(shifted_notes[pitch])[:,0]

            for i in range(len(score_notes[pitch])):
                offset = score_notes[pitch][i][0]
                idx_list = (np.abs(shifted_offsets - offset)).argsort()
                for idx in idx_list:
                    if abs(shifted_notes[pitch][idx][0] - offset) > .5:
                        #print(offset, shifted_notes[pitch][idx][0])
                        break
                    if False:
                        pass
                    else: 
                        shifted_notes[pitch][idx][1] = True
                        score_notes[pitch][i][1] = True # Has been matched <- True
                        score_notes[pitch][i][2] = shifted_notes[pitch][idx][2] # Velocity
                        score_notes[pitch][i][4] = shifted_notes[pitch][idx][3] # Note length of player
                        all_timed_score_vels[score_notes[pitch][i][3]] = shifted_notes[pitch][idx][2] # Velocity in flatlist
                        all_timed_score_lengths[score_notes[pitch][i][3]] = shifted_notes[pitch][idx][3] # Length in flatlist
                        break

        for i in score_notes:
            for ind, j in enumerate(score_notes[i]):
                if not j[1]:
                    j[2] = round(np.sum(all_timed_score_vels[max(0, score_notes[i][ind][3]-3) : min(len(all_timed_score_vels), score_notes[i][ind][3]+4)])/6)
                    #j[1] = True
                    #j[6] = False
                    j[4] = np.sum(all_timed_score_lengths[max(0, score_notes[i][ind][3]-2) : min(len(all_timed_score_vels), score_notes[i][ind][3]+3)])/4
                if j[5] == -1:
                    #print(max(0, score_notes[i][ind][3]-3), min(len(all_timed_score_vels), score_notes[i][ind][3]+3), score_notes[i][ind], i, ind)
                    j[6] = False
                    #j[5] = np.mean(all_timed_score_lengths[max(0, score_notes[i][ind][3]-3) : min(len(all_timed_score_vels), score_notes[i][ind][3]+3)])

        all_notes_and_data = []
        for note, occurances in score_notes.items():
            for occurance in occurances:
                all_notes_and_data.append([note] + occurance)

        all_notes_and_data.sort(key=lambda note_press: (note_press[1], note_press[0])) # Sort by time, then note lowest -> highest

        return shifted_notes, score_notes, all_notes_and_data

# Midi_number, time_when_happens, matched?, velocity, link_to_flatlist(old), note_length_played, note_length_original, got_stop_signal?

for i, row in METADATA.iterrows():
    print(f"Processing {row['midi_performance']}")
    processed_path = os.path.join(PROCESSED_PATH, row["midi_performance"])[:-4] + ".txt"

    shifted, score, all_notes_and_data = parse_midi(row['midi_performance'])

    if not all_notes_and_data:
        continue
    
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    with open(processed_path, "w") as f:
        f.write("\n".join("\t".join(str(x) for x in a) for a in all_notes_and_data))


#parse_midi("Chopin/Scherzos/39/Bult-ItoS05M.mid")