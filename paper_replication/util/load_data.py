import json
import pandas
import random
import copy

# handles getting the features from the processed/aligned data along with extract_features, where the processed data is made in
# process_midi
def prepare_dataset(data_path, metadata_path, columns_with_hist, columns_without_hist, goal_columns, test_data_only = False, train_by_piece = False):
    data = json.load(open(data_path))
    df = pandas.read_csv(metadata_path)

    columns = columns_with_hist + columns_without_hist + [f"-1_{col}" for col in columns_with_hist]

    composers_pieces = {}
    composers_total = {}

    for perf_name in data:
        md = df.loc[df['midi_performance'] == perf_name].values
        if f"{md[0][0]} {md[0][1]}" == "Mozart Fantasie_475":
            continue # Skip this piece, it's out of place

        if md[0][0] not in composers_pieces:
            composers_pieces[md[0][0]] = {}
            composers_total[md[0][0]] = 0
        
        if md[0][1] not in composers_pieces[md[0][0]]:
            composers_pieces[md[0][0]][md[0][1]] = len(data[perf_name]['Note'])
        else:
            composers_pieces[md[0][0]][md[0][1]] += len(data[perf_name]['Note'])
        composers_total[md[0][0]] += len(data[perf_name]['Note'])

    train_pieces, val_pieces, test_pieces = [], [], []

    for composer in composers_pieces:
        pieces = [[], [], []]
        totals = [0, 0, 0]

        num_val_samples = int(0.15 * composers_total[composer])
        num_train_samples = composers_total[composer] - (2 * num_val_samples)
        
        goals = [num_train_samples, num_val_samples, num_val_samples]

        sorted_pieces = sorted(composers_pieces[composer].items(), key=lambda x:x[1], reverse=True)
        random.Random(4).shuffle(sorted_pieces)
        counter = 0
        for piece, length in sorted_pieces:
            for i in range(3):
                if totals[counter] + length <= goals[counter]:
                    totals[counter] += length
                    pieces[counter].append(f"{composer} {piece}")
                    break

                counter += 1
                
                if counter > 2:
                    counter = 0
            else:
                pieces[totals.index(min(totals))].append(f"{composer} {piece}")
                totals[totals.index(min(totals))] += length
        
        train_pieces.extend(pieces[0])
        val_pieces.extend(pieces[1])
        test_pieces.extend(pieces[2])

    train_samples, val_samples, test_samples, train_samples_by_piece, test_samples_by_piece = {}, {}, {}, {}, {}

    for perf_name in data:
        md = df.loc[df['midi_performance'] == perf_name].values
        if f"{md[0][0]} {md[0][1]}" == "Mozart Fantasie_475":
            continue # Skip this piece, it's out of place
        
        data_wanted = {k: v for k, v in data[perf_name].items() if k in columns}
        
        if f"{md[0][0]} {md[0][1]}" in train_pieces:
            train_samples_by_piece[perf_name] = copy.deepcopy(data_wanted)
            if not train_samples:
                train_samples = data_wanted
                continue
            for k in data_wanted:
                train_samples[k] += data_wanted[k]
        elif f"{md[0][0]} {md[0][1]}" in val_pieces:
            if not val_samples:
                val_samples = data_wanted
                continue
            for k in data_wanted:
                val_samples[k] += data_wanted[k]
        elif f"{md[0][0]} {md[0][1]}" in test_pieces:
            test_samples_by_piece[perf_name] = copy.deepcopy(data_wanted) # Otherwise the first piece becomes huge. No idea why, investigate?
            if not test_samples:
                test_samples = data_wanted
                continue
            for k in data_wanted:
                test_samples[k] += data_wanted[k]
        
        else:
            print(f"Missing piece! {md[0][0]} {md[0][1]}")
        
    train_df, val_df, test_df = list(map(pandas.DataFrame, [train_samples, val_samples, test_samples]))
    test_dfs = {piece_name: pandas.DataFrame(samples) for piece_name, samples in test_samples_by_piece.items()}
    train_dfs = {piece_name: pandas.DataFrame(samples) for piece_name, samples in train_samples_by_piece.items()}
    test_gs = {}
    train_gs = {}

    train_targets = pandas.concat([train_df.pop(x) for x in goal_columns], axis=1)
    val_targets = pandas.concat([val_df.pop(x) for x in goal_columns], axis=1)
    test_targets = pandas.concat([test_df.pop(x) for x in goal_columns], axis=1)
    for d in test_dfs:
        test_gs[d] = pandas.concat([test_dfs[d].pop(x) for x in goal_columns], axis=1)
    for d in train_dfs:
        train_gs[d] = pandas.concat([train_dfs[d].pop(x) for x in goal_columns], axis=1)

    print(train_df.shape[0], val_df.shape[0], test_df.shape[0])

    if test_data_only:
        return test_dfs, test_gs
    elif not train_by_piece:
        return train_df, train_targets, val_df, val_targets, test_df, test_targets
    else:
        return train_dfs, train_gs, val_df, val_targets, test_df, test_targets
