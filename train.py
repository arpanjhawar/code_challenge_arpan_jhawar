import os
import glob
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from scipy.stats import skew
import sys
from sklearn import preprocessing


def extract_attributes(data_1, data_2):
    corr_data = np.correlate(data_1, data_2, "full")
    mean = corr_data.mean()
    std = corr_data.std()
    var = np.var(corr_data)
    maximum = corr_data.max()
    sums = corr_data.sum()
    sk = skew(corr_data)
    attri = [mean, std, var, maximum, sums, sk]
    return attri


if __name__ == "__main__":
    data_dir = sys.argv[1]
    filenames = glob.glob(os.path.join(data_dir, "tt*.pkl"))
    attributes = []
    imdbs = []
    shot_end_frames = []
    for fn in filenames:
        x = pickle.load(open(fn, "rb"))
        imdb = x["imdb_id"]
        place = x['place'].numpy()
        cast = x['cast'].numpy()
        action = x['action'].numpy()
        audio = x['audio'].numpy()
        scene_transition_boundary_ground_truth = x['scene_transition_boundary_ground_truth'].numpy()
        shot_end_frame = x['shot_end_frame']
        imdbs.append(imdb)
        shot_end_frames.append(shot_end_frame)
        for p in range(len(place) - 2):
            attr = []
            place_attr = extract_attributes(place[p + 1], place[p + 2])
            cast_attr = extract_attributes(cast[p + 1], cast[p + 2])
            action_attr = extract_attributes(action[p + 1], action[p + 2])
            audio_attr = extract_attributes(audio[p + 1], audio[p + 2])
            attr.extend(place_attr)
            attr.extend(cast_attr)
            attr.extend(action_attr)
            attr.extend(audio_attr)
            attr.append(scene_transition_boundary_ground_truth[p] or scene_transition_boundary_ground_truth[p + 1])
            attr.append(imdb)
            attributes.append(attr)
    df = pd.DataFrame(attributes)
    df[[24]] *= 1
    X, y = df[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]], df[[24]]
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    lr = LogisticRegression(max_iter=10000, class_weight={1: 10, 0: 1})
    lr.fit(X, y)
    predicted = lr.predict(X)
    df.rename(columns={25: 'imdb', 24: 'ground_truth'}, inplace=True)
    df['predicted'] = predicted
    path = sys.argv[1] + '/predicted_data'
    os.mkdir(path)
    save_path = sys.argv[1] + '/predicted_data/'
    for i in range(len(imdbs)):
        d = {}
        data = df.loc[df['imdb'] == imdbs[i]]
        data['predicted'] = data['predicted'].astype(bool)
        data['ground_truth'] = data['ground_truth'].astype(bool)
        d['imdb_id'] = imdbs[i]
        d['shot_end_frame'] = shot_end_frames[i]
        gt = data['ground_truth'].tolist()
        pred = data['predicted'].tolist()
        d['scene_transition_boundary_ground_truth'] = torch.BoolTensor(gt)
        d['scene_transition_boundary_prediction'] = torch.BoolTensor(pred)
        fname = imdbs[i] + '.pkl'
        pickle.dump(d, open(save_path + fname, 'wb'))
