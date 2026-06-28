"""
SAND ALS severity pipeline — SVM-only, debugged and self-contained.
Usage:
  python sand2_svm_fixed.py --dataset "C:\path\to\training" --metadata "C:\path\to\sand_task_1.xlsx" --Nw 0.8 --n_folds 5
Requirements:
  pip install numpy pandas soundfile librosa scipy scikit-learn tqdm
"""

import os
import glob
import argparse
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy import stats
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# Config / constants
# -------------------------
SR_TARGET = 16000
FRAME_WIN = 0.020
FRAME_HOP = 0.010
Nw_default = 0.8
Nsh = 0.1
RANDOM_STATE = 42

# -------------------------
# Audio helpers
# -------------------------
def load_audio(filename, sr=SR_TARGET):
    """Read audio (soundfile) and resample using librosa (compatible with 0.10+)."""
    x, orig_sr = sf.read(filename)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    if orig_sr != sr:
        # librosa 0.10+ signature
        x = librosa.resample(y=x.astype(np.float32), orig_sr=orig_sr, target_sr=sr)
    return x.astype(np.float32), sr

def compute_mfcc_36(x, sr=SR_TARGET):
    hop_length = int(round(FRAME_HOP * sr))
    win_length = int(round(FRAME_WIN * sr))
    # request 13 MFCCs then drop the 0th (energy) -> keep 1..12
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13, n_fft=max(512, win_length*2), hop_length=hop_length)
    if mfcc.shape[0] < 13:
        raise RuntimeError("librosa returned fewer MFCCs than expected")
    mfcc12 = mfcc[1:13, :]                 # (12, n_frames)
    delta = librosa.feature.delta(mfcc12)
    delta2 = librosa.feature.delta(mfcc12, order=2)
    feat = np.concatenate([mfcc12, delta, delta2], axis=0)   # (36, n_frames)
    return feat.T   # (n_frames, 36)

def apply_cmvn(frames):
    mu = np.mean(frames, axis=0)
    sigma = np.std(frames, axis=0)
    sigma[sigma == 0] = 1.0
    return (frames - mu) / sigma

def suprasegmental_from_frames(frames, Nw=Nw_default, Nsh=Nsh):
    hop_s = FRAME_HOP
    win_s = FRAME_WIN
    if Nw <= win_s:
        frames_per_window = 1
    else:
        frames_per_window = int(round((Nw - win_s) / hop_s)) + 1
    shift_per_window = max(1, int(round(Nsh / hop_s)))
    n_frames = frames.shape[0]
    supraseg_feats = []
    start = 0
    while start + frames_per_window <= n_frames:
        win_frames = frames[start:start + frames_per_window, :]
        mean = np.mean(win_frames, axis=0)
        median = np.median(win_frames, axis=0)
        sd = np.std(win_frames, axis=0)
        supraseg_feats.append(np.concatenate([mean, median, sd], axis=0))
        start += shift_per_window
    if len(supraseg_feats) == 0:
        # fallback to global stats
        mean = np.mean(frames, axis=0)
        median = np.median(frames, axis=0)
        sd = np.std(frames, axis=0)
        supraseg_feats.append(np.concatenate([mean, median, sd], axis=0))
    return np.array(supraseg_feats)  # (n_windows, 108)

# -------------------------
# Dataset builder (Excel mapping)
# -------------------------
def build_dataset(root_dir, metadata_path):
    """
    root_dir: folder that contains phonationA, phonationE, ..., rhythmKA, ...
    metadata_path: path to sand_task_1.xlsx with columns ID, Age, Sex, Class (1-5)
    """
    meta_df = pd.read_excel(metadata_path)
    # normalize column names
    meta_df.columns = [c.strip().lower() for c in meta_df.columns]
    id_col = next((c for c in meta_df.columns if 'id' in c), None)
    class_col = next((c for c in meta_df.columns if 'class' in c), None)
    if id_col is None or class_col is None:
        raise ValueError("Excel metadata must contain ID and Class columns")
    id_to_info = {}
    for _, row in meta_df.iterrows():
        key = str(row[id_col]).strip().upper()
        try:
            label = int(row[class_col])
        except Exception:
            label = int(float(row[class_col]))
        id_to_info[key] = {'label': label, 'age': row.get('age') if 'age' in meta_df.columns else None, 'sex': row.get('sex') if 'sex' in meta_df.columns else None}

    entries = []
    for task_dir in sorted(glob.glob(os.path.join(root_dir, '*'))):
        if not os.path.isdir(task_dir):
            continue
        task_name = os.path.basename(task_dir)
        wavs = sorted(glob.glob(os.path.join(task_dir, '*.wav')) + glob.glob(os.path.join(task_dir, '*.WAV')))
        for w in wavs:
            fname = os.path.basename(w)
            base = os.path.splitext(fname)[0].strip().upper()
            subj_id = base.split('_')[0]  # handles names like ID000_phonationA
            if subj_id not in id_to_info:
                # try a slightly different normalization (remove hyphens/extra)
                subj_key = subj_id.replace('-', '').replace(' ', '')
                found = None
                for k in id_to_info.keys():
                    if k.replace('-', '').replace(' ', '') == subj_key:
                        found = k; break
                if found is None:
                    print(f"⚠️ Skipping {fname}: {subj_id} not found in metadata")
                    continue
                subj_id = found
            info = id_to_info[subj_id]
            entries.append({'subject_id': subj_id, 'utterance_path': w, 'label': info['label'], 'task': task_name, 'age': info.get('age'), 'sex': info.get('sex')})
    print(f"✅ Found {len(entries)} utterances across {len(set(e['subject_id'] for e in entries))} subjects.")
    # print class distribution
    counts = defaultdict(int)
    for e in entries:
        counts[e['label']] += 1
    print("Class counts:", dict(counts))
    return entries

# -------------------------
# Per-utterance feature extraction
# -------------------------
def extract_utterance_supraseg(utt_path, Nw=Nw_default):
    x, sr = load_audio(utt_path)
    frames = compute_mfcc_36(x, sr)
    frames = apply_cmvn(frames)
    return suprasegmental_from_frames(frames, Nw=Nw, Nsh=Nsh)

# -------------------------
# Evaluate pipeline (SVM-only, safe defaults)
# -------------------------
def evaluate_pipeline(entries, Nw=0.8, n_folds=5, subsample_windows=None):
    """
    subsample_windows: if int, randomly sample up to that many training windows to speed SVM.
    """
    if len(entries) == 0:
        raise ValueError("No entries found. Check dataset and metadata paths.")
    subjects = np.array(sorted(set(e['subject_id'] for e in entries)))
    # create label vector per subject (take first occurrence)
    labels = np.array([next(e['label'] for e in entries if e['subject_id'] == s) for s in subjects])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    all_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(subjects, labels)):
        start_time = time.time()
        train_subj = set(subjects[train_idx])
        test_subj = set(subjects[test_idx])
        print(f"\n===== Fold {fold+1}/{n_folds} =====")

        X_train, y_train, X_test, y_test = [], [], [], []

        # extract features with progress bar
        for e in tqdm(entries, desc="Extracting features", unit="file"):
            try:
                supr = extract_utterance_supraseg(e['utterance_path'], Nw)
            except Exception as ex:
                print(f"⚠️ Skipping {e['utterance_path']} due to error: {ex}", flush=True)
                continue
            if e['subject_id'] in train_subj:
                for s in supr:
                    X_train.append(s)
                    y_train.append(e['label'])
            elif e['subject_id'] in test_subj:
                for s in supr:
                    X_test.append(s)
                    y_test.append(e['label'])

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        print(f"✅ Feature extraction done. Train windows: {X_train.shape[0]}, Test windows: {X_test.shape[0]}")

        # optional subsample to keep SVM tractable
        if subsample_windows is not None and X_train.shape[0] > subsample_windows:
            idx = np.random.RandomState(RANDOM_STATE).choice(X_train.shape[0], subsample_windows, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]
            print(f"🔽 Subsampled training windows to {subsample_windows}")

        # train/val split
        if len(np.unique(y_train)) < 2:
            print("⚠️ Not enough classes in training windows for this fold; skipping.")
            continue
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE)

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        # light grid by default to avoid huge runs
        param_grid = {'C': [1, 10], 'gamma': ['scale']}
        grid = GridSearchCV(SVC(kernel='rbf', decision_function_shape='ovr'), param_grid, cv=2, n_jobs=-1, verbose=2)
        print("🧠 Training SVM (GridSearch)...", flush=True)
        grid.fit(X_tr_s, y_tr)
        best_svm = grid.best_estimator_
        print(f"✅ Best SVM params: {grid.best_params_}", flush=True)

        y_pred = best_svm.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        print(f"Fold {fold+1}/{n_folds}: SVM accuracy = {acc:.3f}")
        print(classification_report(y_test, y_pred, digits=3))
        all_results.append(acc)

        fold_time = (time.time() - start_time) / 60.0
        print(f"⏱ Fold {fold+1} finished in {fold_time:.2f} minutes\n", flush=True)

    if len(all_results) > 0:
        print("Average SVM accuracy:", float(np.mean(all_results)))
    else:
        print("No folds completed.")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAND ALS severity SVM pipeline")
    parser.add_argument('--dataset', required=True, help='Path to training folder (contains phonation*/rhythm* subfolders)')
    parser.add_argument('--metadata', required=True, help='Path to sand_task_1.xlsx')
    parser.add_argument('--Nw', type=float, default=0.8, help='Suprasegmental window (s)')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of subject-level CV folds')
    parser.add_argument('--subsample', type=int, default=20000, help='Subsample windows for SVM (set 0 to disable)')
    args = parser.parse_args()

    subs = None if args.subsample == 0 else args.subsample
    entries = build_dataset(args.dataset, args.metadata)
    evaluate_pipeline(entries, Nw=args.Nw, n_folds=args.n_folds, subsample_windows=subs)
