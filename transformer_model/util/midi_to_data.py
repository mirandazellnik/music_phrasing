from mido import MidiFile

x = "/stash/tlab/theom_intern/midi_data/little_pieces_4_(c)oguri.mid"

def data_from_midi(path):
    mid = MidiFile(path)
    notes = []
    vols = []
    for i, track in enumerate(mid.tracks):
        if track.name != "PIANO":
            continue
        for msg in track:
            if msg.type != "note_on":
                continue
            if msg.velocity == 0:
                continue
            #print(msg.note)
            notes.append(msg.note)
            vols.append(msg.velocity)
    return notes, vols
