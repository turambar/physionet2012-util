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


try:
    os.makedirs('raw')
except:
    pass
download_set_a('raw')
download_set_b('raw')
download_set_a_targets('raw')
try:
    os.makedirs('prepared')
except:
    pass
if os.path.isfile(os.path.join('prepared', 'transforms.json')):
    with open(os.path.join('prepared', 'transforms.json'), 'r') as f:
        transforms = json.load(f)
else:
    stats = compute_statistics('raw')
    with open(os.path.join('prepared', 'statistics.json'), 'w') as f:
        json.dump(stats, f, indent=1, sort_keys=True)
    transforms = convert_stats_to_rescale_transform(stats)
    with open(os.path.join('prepared', 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=1, sort_keys=True)

#seta = set([ int(fn.split('/')[-1].replace('.txt', '')) for fn in glob.glob(os.path.join('set-a', '*.txt')) ])
targets = read_and_prep_targets('raw')
#train, test, invalid = generate_splits(targets, train_frac=0.75)
# seta = seta - train
# seta = seta - test
# invalid.update(seta)
# np.random.shuffle(list(train))

for dirname in [ 'sequence', 'resampled', 'features', 'mortality', 'los', 'los_bucket', 'survival', 'survival_bucket' ]:
    try:
        os.makedirs(os.path.join('prepared', dirname))
    except:
        pass
for set_letter in ['a', 'b']:
    for fn in tqdm(glob.glob(os.path.join('raw', 'set-' + set_letter, '*.txt'))[:10]):
        recordid = int(fn.split('/')[-1].replace('.txt', ''))
        record = DataFrame.from_csv(fn, index_col=None)
        record.drop_duplicates(subset=['Time', 'Parameter'], inplace=True)
        record = pivot_record(record, var_names=transforms.keys())
        record = apply_transforms(record, transforms)
        record = convert_time(record)
        record = add_missing_columns(record, transforms.keys())
        if 'Cholesterol' not in record.columns:
            assert(False)
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
        # features = extract_features(record)

        record.to_csv(os.path.join('prepared', 'sequence', '{}.csv'.format(recordid)), index=False)
        resampled.to_csv(os.path.join('prepared', 'resampled', '{}.csv'.format(recordid)), index=False)
        # with open(os.path.join('features', '{}.csv'.format(recordid)), 'w') as fout:
        #     fout.write(','.join([ str(x) for x in features ]) + '\n')

        if recordid in targets.index:
            with open(os.path.join('prepared', 'mortality', '{}.csv'.format(recordid)), 'w') as f:
                f.write(str(targets.Mortality[recordid]) + '\n')
            with open(os.path.join('prepared', 'los', '{}.csv'.format(recordid)), 'w') as f:
                f.write(str(targets.Los[recordid]) + '\n')
            with open(os.path.join('prepared', 'los_bucket', '{}.csv'.format(recordid)), 'w') as f:
                f.write(str(targets.LosClass[recordid]) + '\n')
            with open(os.path.join('prepared', 'survival', '{}.csv'.format(recordid)), 'w') as f:
                f.write(str(targets.Survival[recordid]) + '\n')
            with open(os.path.join('prepared', 'survival_bucket', '{}.csv'.format(recordid)), 'w') as f:
                f.write(str(targets.SurvivalClass[recordid]) + '\n')
