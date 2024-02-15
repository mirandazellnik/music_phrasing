""" For applying the outputs of the model into the original parsed MIDI -> txt files, to create the more realistic files. """

import os
import json

import numpy as np
import pandas
import mido


ASAP_PATH = "/stash/tlab/theom_intern/midi_data/asap-dataset-master"
PROCESSED_PATH = "/stash/tlab/theom_intern/midi_data/asap-dataset-processed"

METADATA = pandas.read_csv(os.path.join(ASAP_PATH, "metadata.csv"))
ASAP_ANNOTATIONS = json.load(open(os.path.join(ASAP_PATH, "asap_annotations.json")))

def apply_outputs(path_to_txt, path_to_save, velocities, lengths):
    mf = mido.MidiFile()
    track = mido.MidiTrack()
    mf.tracks.append(track)

    #track.append(mido.MetaMessage('time_signature', numerator=3, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))

    track.append(mido.Message('program_change', program=0, time=0))
    track.append(mido.Message(type="control_change", channel=0, control=7, value=100, time=0)) # Volume
    track.append(mido.Message(type="control_change", channel=0, control=91, value=0, time=0)) # "Depth 1" 
    track.append(mido.Message(type="control_change", channel=0, control=93, value=0, time=0)) # "Depth 2"

    with open(path_to_txt) as f:
        lines = f.read().splitlines()
    events = []
    for i, line in enumerate(lines):
        note, offset, _, _, _, _, length, _ = line.split('\t')
        note, offset, length = int(note), float(offset), float(length)
        events.append([offset, note, round(float(velocities[i])*3) + 65])
        events.append([offset + float(lengths[i]), note, 0])
    
    events.sort(key = lambda x: x[0])

    offset = 0
    for event in events:
        time = event[0] - offset
        track.append(mido.Message(type="note_on", channel=0, note=event[1], velocity=event[2], time=round(mf.ticks_per_beat * time * 2)))
        offset = event[0]
    
    mf.save(path_to_save)