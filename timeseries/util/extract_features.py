import os
import json
import mido

import pandas
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

saving = True
load = False

def moving_average(a, n):
    b = []
    for i in range(len(a)):
        new_slice = a[max(0, i-n) : i+n+1]
        b.append(sum(new_slice)/len(new_slice))
    return b

ASAP_PATH = "/stash/tlab/theom_intern/midi_data/asap-dataset-master"
PROCESSED_PATH = "/stash/tlab/theom_intern/midi_data/asap-dataset-processed"

METADATA = pandas.read_csv(os.path.join(ASAP_PATH, "metadata.csv"))
ASAP_ANNOTATIONS = json.load(open(os.path.join(ASAP_PATH, "asap_annotations.json")))

dists = [".01", ".05", ".10", ".50", "1.0", "2.0", "4.0"]


cols = ["Note", "Time", "Len_P", "Len_M", "D_A", "Exact", "Exact_L", "Exact_H", "Motion", "BPM", "Len/BPM", "Len_Ratio", "Time_From", "Time_To"] + [f"W" + dist for dist in dists] + [f"B" + dist for dist in dists] + [f"A" + dist for dist in dists] + ["Velo", "Macro", "Micro"]
seq_total = {cname:[] for cname in cols}
seq_by_piece = {}

if not load:
    for count, row in METADATA.iterrows():

        if row["midi_performance"][:5] not in ["Bach/", "Haydn", "Mozar"]:
            continue

        processed_path = os.path.join(PROCESSED_PATH, row["midi_performance"])[:-4] + ".txt"

        if not ASAP_ANNOTATIONS[row["midi_performance"]]["score_and_performance_aligned"]:
            continue

        try:
            with open(processed_path, "r") as f:
                lines = f.read().splitlines()
        except:
            print(f"MISSING! {row['midi_performance']}")
            continue

        perf_path = os.path.join(ASAP_PATH, row["midi_performance"])
        score_path = os.path.join(os.path.dirname(perf_path), "midi_score.mid")
        mf = mido.MidiFile(score_path)

        for elem in mf:
            if elem.type == 'set_tempo':
                tempo = 60000000/elem.tempo
                break
        else:
            print(f"MISSING TEMPO {row['midi_performance']}")

        lines = [line.split('\t') for line in lines]
        lists = np.transpose(lines)

        try:
            lists = lists[[0,1,5,6,3,]]
        except IndexError:
            print(f"REDO ME!! {row['midi_performance']}")
            continue
        lists = [list(map(float, l)) for l in lists]

        new_starts = []
        new_ends = []


        down_while = []
        
        within = []

        ahead_only = []
        behind_only = []
        exact = []
        exact_l = []
        exact_h = []

        motion = []

        length_ratio = []

        tempo_bpm = []
        len_tempo = []

        time_from_last = []
        time_to_next = []


        for s, length in zip(lists[1], lists[3]):
            new_starts.append(s)
            new_ends.append(s + length)
        

        for n in range(len(new_starts)):
            down_while.append(0)
            ahead_only.append([0]*7)
            behind_only.append([0]*7)
            exact.append(0)
            exact_l.append(0)
            exact_h.append(0)

            motion.append(0)
            length_ratio.append(1)
            tempo_bpm.append(tempo)
            len_tempo.append(lists[3][n]/tempo)

            time_from_last.append(0)
            time_to_next.append(0)


            for pos, j in enumerate(new_starts[max(n-100, 0):n]):
                if j < new_starts[n]:
                    time_from_last[-1] = new_starts[n] - j
                    length_ratio[-1] = lists[3][n]/lists[3][pos]
                    motion[-1] = lists[0][n] - lists[0][pos]
                if new_starts[n] < j < new_ends[n]:
                    down_while[-1] += 1
                if new_starts[n] == j:
                    exact[-1] += 1
                    exact_l[-1] += 1
                    continue
                if abs(new_starts[n] - j) < .01:
                    behind_only[-1][0] += 1
                if abs(new_starts[n] - j) < .05:
                    behind_only[-1][1] += 1
                if abs(new_starts[n] - j) < .1:
                    behind_only[-1][2] += 1
                if abs(new_starts[n] - j) < .5:
                    behind_only[-1][3] += 1
                if abs(new_starts[n] - j) < 1:
                    behind_only[-1][4] += 1
                if abs(new_starts[n] - j) < 2:
                    behind_only[-1][5] += 1
                if abs(new_starts[n] - j) < 4:
                    behind_only[-1][6] += 1
            
            for j in new_starts[n + 100 : n : -1]:
                if new_starts[n] < j:
                    time_to_next[-1] = j - new_starts[n]
                if new_starts[n] < j < new_ends[n]:
                    down_while[-1] += 1
                if new_starts[n] == j:
                    exact[-1] += 1
                    exact_h[-1] += 1
                    continue
                if abs(new_starts[n] - j) < .01:
                    ahead_only[-1][0] += 1
                if abs(new_starts[n] - j) < .05:
                    ahead_only[-1][1] += 1
                if abs(new_starts[n] - j) < .1:
                    ahead_only[-1][2] += 1
                if abs(new_starts[n] - j) < .5:
                    ahead_only[-1][3] += 1
                if abs(new_starts[n] - j) < 1:
                    ahead_only[-1][4] += 1
                if abs(new_starts[n] - j) < 2:
                    ahead_only[-1][5] += 1
                if abs(new_starts[n] - j) < 4:
                    ahead_only[-1][6] += 1
            
            within.append([ahead_only[-1][i]+behind_only[-1][i] for i in range(7)])
        

        within = np.transpose(within).tolist()
        ahead_only = np.transpose(ahead_only).tolist()
        behind_only = np.transpose(behind_only).tolist()

        maxtime = lists[1][-1]
        lists[1] = [x/maxtime for x in lists[1]] # Scale time from 0-1 for start-finish

        lists = lists[:-1] + [down_while, exact, exact_l, exact_h, motion, tempo_bpm, len_tempo, length_ratio, time_from_last, time_to_next] + within + behind_only + ahead_only + [lists[-1]]

        averaged_vol = moving_average(lists[-1], 4)
        averaged_vol = moving_average(averaged_vol, 2)

        lists = [l for l in lists] + [averaged_vol]

        micro_vol = np.subtract(lists[-2], lists[-1]).tolist()

        lists += [micro_vol]

        if saving:
                
            seq = {cname:[] for cname in cols}
            for i, v in enumerate(lists):
                seq[cols[i]] += v
            #df = pandas.DataFrame(seq)
            #df.to_csv(os.path.join(PROCESSED_PATH, "last_piece.tsv"), sep="\t")
            
            seq_by_piece[row["midi_performance"]] = seq

            for i, v in enumerate(lists):
                seq_total[cols[i]] += v
        else:
            for i, v in enumerate(lists):
                seq_total[cols[i]] += v
            
                
        if count % 10 == 0:
            print(count)
        


if saving:

    if not load:
        shifted_by_piece = {}

        for name, piece in seq_by_piece.items():
            timeshifted = {key: val[1:] for key, val in piece.items()}
            timeshifted.update({f"-1_{key}": piece[key][:-1] for key in ["Note","Exact_L","Exact_H","Motion","Micro","Macro"]})
            shifted_by_piece[name] = timeshifted


        with open(os.path.join(PROCESSED_PATH, "shifted_by_piece.json"), "w") as outfile:
            json.dump(shifted_by_piece, outfile)
        
    with open(os.path.join(PROCESSED_PATH, "shifted_by_piece.json")) as infile:
        x = json.load(infile)

        f = {}
        cols = ["Note", "Exact_L", "Exact_H", "Motion"] + ["Len_M", "W.50", "B.10", "B.50", "A.10", "A.50", "-1_Micro"] + [f"-1_{col}" for col in ["Note", "Exact_L", "Exact_H", "Motion"]] + ["Micro"]

        for i, key in enumerate(x):
            if i == 0:
                f = x[key]
            else:
                for k in x[key]:
                    f[k] += x[key][k]
        
        f = {k:v for k,v in f.items() if k in cols}
        
        print(len(f["Note"]), len(f))

        f = pandas.DataFrame(f)
        
        for col in f:
            sn.displot(f[col])
            plt.show()

        raise ValueError
        df = pandas.DataFrame(x)
        
        corr_matrix = df.corr()
        sn.heatmap(corr_matrix, annot=True, xticklabels=1, yticklabels=1)
        plt.show()
    
else:
    print(max(seq_total["Micro"]), min(seq_total["Micro"]))
    print(max(seq_total["Macro"]), min(seq_total["Macro"]))
    df = pandas.DataFrame(seq_total)
    corr_matrix = df.corr()
    sn.heatmap(corr_matrix, annot=True, xticklabels=1, yticklabels=1)
    plt.show()