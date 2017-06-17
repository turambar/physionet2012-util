import glob
import numpy as np
import os
import pandas as pd
import requests
import tarfile

from collections import defaultdict, Counter
from csv import DictReader
from pandas import DataFrame
from tqdm import tqdm


def download_set_a(path='.'):
    _download_set('a', path)


def download_set_b(path='.'):
    _download_set('b', path)


def download_set_a_targets(path='.'):
    dest = os.path.join(path, 'Outcomes-a.txt')
    if not os.path.isfile(dest):
        url = 'https://physionet.org/challenge/2012/Outcomes-a.txt'
        response = requests.get(url, stream=True)
        with open(dest, 'wb') as f:
            for data in tqdm(response.iter_content()):
                f.write(data)


def _download_set(letter, path='.'):
    dest = os.path.join(path, 'set-{}.tar.gz'.format(letter))
    if not os.path.isfile(dest):
        url = 'https://physionet.org/challenge/2012/set-{}.tar.gz'.format(letter)
        response = requests.get(url, stream=True)
        with open(dest, 'wb') as f:
            for data in tqdm(response.iter_content()):
                f.write(data)
    dir = os.path.join(path, 'set-{}'.format(letter))
    if not os.path.isdir(dir):
        tar = tarfile.open(dest, "r:gz")
        tar.extractall()
        tar.close()


def compute_statistics(path='.'):
    data = defaultdict(list)
    for patient_set in [ 'set-a', 'set-b']:
        for fn in tqdm(glob.glob(os.path.join(path, patient_set, '*.txt'))):
            reader = DictReader(open(fn))
            for record in reader:
                if record['Parameter'] != 'RecordID':
                    try:
                        value = np.float(record['Value'])
                        if value != -1.0:
                            data[record['Parameter']].append(value)
                    except:
                        print('Could not convert to float:' + str(record))
    stats = defaultdict(list)
    for var_name in data:
        values = data[var_name]
        if var_name == 'Gender' or var_name == 'ICUType' or var_name == 'MechVent':
            stats[var_name] = Counter([ np.int(v) for v in values ])
        else:
            stats[var_name] = list(np.percentile(values, [0, 1, 50, 99, 100]))
            stats[var_name].extend([ np.mean(values), np.std(values), len(values) ])
    return stats


def convert_stats_to_rescale_transform(stats):
   transforms = {}
   for var_name in stats:
       if var_name == 'Gender' or var_name == 'ICUType':
           transforms[var_name] = {}
           for i, val in enumerate(sorted(stats[var_name])):
               transforms[var_name][val] = ( var_name + str(val), i)
       elif var_name == 'MechVent':
           transforms[var_name] = {'min': 0, 'normal': 0, 'max': 1}
       else:
           transforms[var_name] = { 'min': stats[var_name][1], 'normal': stats[var_name][2], 'max': stats[var_name][3] }
   return transforms


def _parse_time_string_to_hours(s):
    hours, mins = s.split(':')
    return float(hours) + float(mins)/60


def pivot_record(record, var_names=None):
    record = record.pivot(index='Time', columns='Parameter', values='Value').reset_index()

    del record['RecordID']
    record['Gender'] = record.Gender.iloc[0] if 'Gender' in record.columns else np.nan
    record['ICUType'] = record.ICUType.iloc[0] if 'ICUType' in record.columns else np.nan
    record['Age'] = record.Age.iloc[0] if 'Age' in record.columns else np.nan
    record['Weight'] = record.Weight.iloc[0] if 'Weight' in record.columns else np.nan
    record['Height'] = record.Weight.iloc[0] if 'Height' in record.columns else np.nan
    record.replace(-1, np.nan, inplace=True)

    if var_names is not None:
        for var_name in var_names:
            if var_name not in record.columns:
                record[var_name] = np.nan
    return record


def get_missing_indicators(record):
    missing = record.isnull().astype(int)
    cols = list(missing.columns)
    for col in cols:
        if col.startswith('Gender') or col.startswith('ICUType') or col.startswith('Elapsed'):
            del missing[col]
    cols = list(missing.columns)
    cols.remove('Time')
    missing.rename_axis(dict(zip(cols, [ c + 'Missing' for c in cols ])), axis=1, inplace=True)
    missing['Time'] = record['Time'].copy()
    return missing


def apply_transforms(record, transforms):
    gender = record['Gender'].iloc[0]
    del record['Gender']
    gender = transforms['Gender'][gender][1] if gender in transforms['Gender'] else -1
    for nm, idx in transforms['Gender'].values():
        record[nm] = 1 if idx == gender else 0
    icutype = record['ICUType'].iloc[0]
    del record['ICUType']
    icutype = transforms['ICUType'][icutype][1] if icutype in transforms['ICUType'] else -1
    for nm, idx in transforms['ICUType'].values():
        record[nm] = 1 if idx == icutype else 0
    for var_name in transforms:
        if var_name == 'Gender' or var_name == 'ICUType':
            continue
        if var_name in record:
            record[var_name] = record[var_name].apply(lambda x: (x - transforms[var_name]['min']) / (transforms[var_name]['max'] - transforms[var_name]['min']))
            record[var_name][record[var_name] < 0] = 0
            record[var_name][record[var_name] > 1] = 1
    return record


def convert_time(record):
    record['Time'] = record.Time.apply(_parse_time_string_to_hours)
    elapsed = np.zeros((record.Time.shape[0]), )
    elapsed[1:] = record.Time.iloc[1:].values - record.Time.iloc[:-1].values
    record['Elapsed'] = elapsed
    record['Time'] = record.Time.apply(lambda h: pd.Timedelta('{} hours'.format(h)))
    return record


def resample_hourly(record):
    resampled = record.resample('1H', closed='left', label='left', on='Time').mean()
    del resampled['Elapsed']
    return resampled.reset_index()


def impute_missing_values(record, transforms=None):
    if transforms is not None:
        for var_name in transforms:
            if var_name != 'Gender' and var_name != 'ICUType' and var_name not in record:
                record[var_name] = np.nan
        for var_name in record.columns:
            if var_name in transforms and np.isnan(record[var_name].iloc[0]):
                val = (transforms[var_name]['normal'] - transforms[var_name]['min']) / (transforms[var_name]['max'] - transforms[var_name]['min'])
                record[var_name].set_value(0, val)
    record.ffill(inplace=True)
    return record


def reorder_columns(record):
    cols = sorted(record.columns)
    if 'Elapsed' in cols:
        cols.remove('Elapsed')
        cols.insert(0, 'Elapsed')
    if 'Time' in cols:
        cols.remove('Time')
        cols.insert(0, 'Time')
    return record[cols]


def bucket_los(los):
    if los <= 2:
        return 0
    elif los > 2 and los <= 3:
        return 1
    elif los > 3 and los <= 4:
        return 2
    elif los > 4 and los <= 5:
        return 3
    elif los > 5 and los <= 6:
        return 4
    elif los > 6 and los <= 7:
        return 5
    elif los > 7 and los <= 14:
        return 6
    else:
        return 7


def bucket_survival(survival):
    if survival >= 0 and survival <= 7:
        return 0
    elif survival > 7 and survival <= 30:
        return 1
    elif survival > 30 and survival <= 365:
        return 2
    elif survival > 365:
        return 3
    else: # including -1
        return 4


def read_and_prep_targets(path='.'):
    targets = DataFrame.from_csv(os.path.join(path, 'Outcomes-a.txt'))
    surv = targets['Survival']
    targets['Survival'][surv == 0] = -1
    targets['Survival'][surv == 1] = targets['Length_of_stay'][surv == 1]
    targets['SurvivalClass'] = targets['Survival'].apply(bucket_survival)
    targets['LosClass'] = targets['Length_of_stay'].apply(bucket_los)
    targets['Los'] = targets['Length_of_stay']
    targets['Mortality'] = targets['In-hospital_death']
    return targets


def generate_splits(targets, train_frac=0.75):
    seta = set(targets.index)
    invalid = set(targets.index[targets.Los < 0])
    mort = list(targets.index[(targets.Mortality == 1) & (targets.Los >= 0)])
    live = list(targets.index[(targets.Mortality == 0) & (targets.Los >= 0)])
    test = set()
    test.update(np.random.choice(mort, int(np.ceil(train_frac * len(mort))), replace=False))
    test.update(np.random.choice(live, int(np.floor(train_frac * len(live))), replace=False))
    train = seta - test
    train = train - invalid
    return train, test, invalid


def extract_features(record):
    tm = record[record.columns[:2]]
    msmt_cols = []
    for col in record.columns[2:]:
        if not col.endswith('Missing') and not col.startswith('Gender') and not col.startswith(
                'ICUType') and not col.startswith('Age') and not col.startswith('Weight') and not col.startswith(
                'Height'):
            msmt_cols.append(col)
    missing_cols = []
    for col in record.columns[2:]:
        if col.endswith('Missing') and not col.startswith('Gender') and not col.startswith(
                'ICUType') and not col.startswith('Age') and not col.startswith('Weight') and not col.startswith(
                'Height'):
            missing_cols.append(col)
    msmts = record[msmt_cols]
    # missing = (1 - record[missing_cols]).replace(0, np.nan)
    features = [np.nanmin(msmts, axis=0),
                np.nanstd(msmts, axis=0)]
    features.extend(np.nanpercentile(msmts, [0, 25, 50, 75, 100], axis=0))
    A, _, _, _ = np.linalg.lstsq(np.hstack([tm.Time.values[:, None], np.ones((tm.Time.shape[0], 1))]), msmts.values)
    features.extend(A)
    features.append(record.Age[0])
    features.append(record.Weight[0])
    features.append(record.Height[0])
    features.extend(record[[col for col in record.columns if col.startswith('Gender')]].iloc[0])
    features.extend(record[[col for col in record.columns if col.startswith('ICUType')]].iloc[0])
    features = np.hstack(features)
    return features
