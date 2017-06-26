import glob
import json
import numpy as np
import os
import pandas as pd
import requests
import tarfile

from collections import defaultdict, Counter
from csv import DictReader
from pandas import DataFrame
from tqdm import tqdm

from physionet2012.data import *


download_set_a()
download_set_b()
download_set_a_targets()
stats = compute_statistics()
with open('statistics.json', 'w') as f:
    json.dump(stats, f, indent=1, sort_keys=True)
transforms = convert_stats_to_rescale_transform(stats)
with open('transforms.json', 'w') as f:
    json.dump(transforms, f, indent=1, sort_keys=True)

seta = set([ int(fn.split('/')[-1].replace('.txt', '')) for fn in glob.glob(os.path.join('set-a', '*.txt')) ])
targets = read_and_prep_targets('.')
train, test, invalid = generate_splits(targets, train_frac=0.75)
seta = seta - train
seta = seta - test
invalid.update(seta)
np.random.shuffle(list(train))
record_no = 0

for dirname in [ 'sequence', 'resampled', 'features', 'mortality', 'los', 'los_bucket', 'survival', 'survival_bucket' ]:
    try:
        os.makedirs(dirname)
    except:
        pass
for records in [ train, test, invalid ]:
    for pid in tqdm(records):
        fn = os.path.join('./set-a', str(pid) + '.txt')
        record = DataFrame.from_csv(fn, index_col=None)
        record.drop_duplicates(subset=['Time', 'Parameter'], inplace=True)
        record = pivot_record(record, var_names=transforms.keys())
        record = apply_transforms(record, transforms)
        record = convert_time(record)
        missing = get_missing_indicators(record)

        resampled = resample_hourly(record)
        missing_resampled = get_missing_indicators(resampled)

        record = impute_missing_values(record, transforms)
        resampled = impute_missing_values(resampled, transforms)

        record = record.merge(missing, left_on='Time', right_on='Time')
        resampled = resampled.merge(missing_resampled, left_on='Time', right_on='Time')
        record['Time'] = record['Time'].apply(lambda d: d.total_seconds()/60/60)
        del resampled['Time']

        record = reorder_columns(record)
        resampled = reorder_columns(resampled)
        features = extract_features(record)

        record.to_csv(os.path.join('sequence', '{}.csv'.format(record_no)), index=False)
        resampled.to_csv(os.path.join('resampled', '{}.csv'.format(record_no)), index=False)
        with open(os.path.join('features', '{}.csv'.format(record_no)), 'w') as fout:
            fout.write(','.join([ str(x) for x in features ]) + '\n')

        with open(os.path.join('mortality', '{}.csv'.format(record_no)), 'w') as f:
            f.write(str(targets.Mortality[pid]) + '\n')
        with open(os.path.join('los', '{}.csv'.format(record_no)), 'w') as f:
            f.write(str(targets.Los[pid]) + '\n')
        with open(os.path.join('los_bucket', '{}.csv'.format(record_no)), 'w') as f:
            f.write(str(targets.LosClass[pid]) + '\n')
        with open(os.path.join('survival', '{}.csv'.format(record_no)), 'w') as f:
            f.write(str(targets.Survival[pid]) + '\n')
        with open(os.path.join('survival_bucket', '{}.csv'.format(record_no)), 'w') as f:
            f.write(str(targets.SurvivalClass[pid]) + '\n')

        record_no += 1

for fn in tqdm(glob.glob(os.path.join('set-b', '*.txt'))):
    pid = int(fn.split('/')[-1].replace('.txt', ''))
    record = DataFrame.from_csv(fn, index_col=None)
    record.drop_duplicates(subset=['Time', 'Parameter'], inplace=True)
    record = pivot_record(record, var_names=None) #transforms.keys())
    record = apply_transforms(record, transforms)
    record = convert_time(record)
    missing = get_missing_indicators(record)

    resampled = resample_hourly(record)
    missing_resampled = get_missing_indicators(resampled)

    record = impute_missing_values(record, transforms)
    resampled = impute_missing_values(resampled, transforms)

    record = record.merge(missing, left_on='Time', right_on='Time')
    resampled = resampled.merge(missing_resampled, left_on='Time', right_on='Time')
    record['Time'] = record['Time'].apply(lambda d: d.total_seconds()/60/60)
    del resampled['Time']

    record = reorder_columns(record)
    resampled = reorder_columns(resampled)

    record.to_csv(os.path.join('sequence', '{}.csv'.format(record_no)), index=False)
    resampled.to_csv(os.path.join('resampled', '{}.csv'.format(record_no)), index=False)

    record_no += 1
