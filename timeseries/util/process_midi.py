""" For processing the ASAP Dataset. """

import os
import json
import pandas
import music21

ASAP_PATH = "/stash/tlab/theom_intern/midi_data/asap-dataset-master"
METADATA = pandas.read_csv(os.path.join(ASAP_PATH, "metadata.csv"))
ASAP_ANNOTATIONS = json.load(open(os.path.join(ASAP_PATH, "asap_annotations.json")))


def parse_midi(path=None, id=None):
    if path:
        perf_path = os.path.join(ASAP_PATH, path)
        perf_ann_path = os.path.splitext(perf_path)[0]+'_annotations.txt'

        score_path = os.path.join(os.path.dirname(perf_path), "midi_score.mid")
        score_ann_path = os.path.splitext(score_path)[0]+'_annotations.txt'

        with open(perf_ann_path) as f:
            perf_ann = [[x[0], x[2]] for x in f.read().splitlines()]
        with open(score_ann_path) as f:
            score_ann = [[x[0], x[2]] for x in f.read().splitlines()]
        
        
        perf_stream = music21.converter.parse(perf_path, quantizePost=False)
        score_stream = music21.converter.parse(score_path, quantizePost=False)

        


        



parse_midi("Chopin/Scherzos/39/Bult-ItoS05M.mid")