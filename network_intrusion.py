import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.decomposition import PCA

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

print("==========================================")
print("1. LOADING DATA & DISCOVERING PATHS")
print("==========================================")

train_files = glob.glob("/kaggle/input/**/train.csv", recursive=True)
test_files = glob.glob("/kaggle/input/**/test.csv", recursive=True)
sample_files = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)

if not train_files or not test_files or not sample_files:
    train_path, test_path, sample_path = 'train.csv', 'test.csv', 'sample_submission.csv'
else:
    train_path, test_path, sample_path = train_files[0], test_files[0], sample_files[0]

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

print(f"Train Shape: {train.shape} | Test Shape: {test.shape}")

TARGET = "Target"
ID = "id"
CLASSES = ["Benign", "Availability", "Reconnaissance", "Spoofing"]

le_target = LabelEncoder()
train[TARGET] = le_target.fit_transform(train[TARGET].astype(str))

print("==========================================")
print("2. ELITE FEATURE ENGINEERING & TRANSFORMATION")
print("==========================================")

df_all = pd.concat([train.drop(columns=[TARGET]), test], axis=0).reset_index(drop=True)

# Clean numeric infinities and missing values
num_cols = df_all.select_dtypes(include=np.number).columns
for col in num_cols:
    df_all[col] = df_all[col].replace([np.inf, -np.inf], np.nan)
    df_all[col] = df_all[col].fillna(df_all[col].median())

# Advanced Domain Network Traffic Interactions
if 'fwd_pkts_tot' in df_all.columns and 'bwd_pkts_tot' in df_all.columns:
    df_all['total_pkts'] = df_all['fwd_pkts_tot'] + df_all['bwd_pkts_tot']
    df_all['pkt_ratio'] = df_all['fwd_pkts_tot'] / (df_all['bwd_pkts_tot'] + 1e-5)
    df_all['fwd_pkt_density'] = df_all['fwd_pkts_tot'] / (df_all['flow_duration'] + 1e-5)

if 'flow_duration' in df_all.columns:
    df_all['duration_log'] = np.log1p(df_all['flow_duration'])

if 'payload_bytes_per_second' in df_all.columns:
    df_all['payload_bytes_log'] = np.log1p(df_all['payload_bytes_per_second'])

# Exclude ID column from categorical features
cat_cols = [c for c in df_all.select_dtypes(include=["object", "category", "string"]).columns if c != ID]

for col in cat_cols:
    df_all[col] = df_all[col].astype(str).fillna("__MISSING__")
    freq_map = df_all[col].value_counts(normalize=True)
    df_all[f'{col}_freq'] = df_all[col].map(freq_map)
    le_cat = LabelEncoder()
    df_all[col] = le_cat.fit_transform(df_all[col])

# PCA Features to capture latent linear manifolds in network traffic
scaler = StandardScaler()
num_scaled = scaler.fit_transform(df_all[num_cols])
pca = PCA(n_components=min(5, len(num_cols)), random_state=42)
pca_features = pca.fit_transform(num_scaled)
for i in range(pca_features.shape[1]):
    df_all[f'pca_{i+1}'] = pca_features[:, i]

X = df_all.iloc[:len(train)].drop(columns=[ID])
y = train[TARGET]
X_test = df_all.iloc[len(train):].drop(columns=[ID])

# Bayesian Smoothed Out-of-Fold Target Encoding for Categoricals
print("-> Computing Out-of-Fold Target Encoding...")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for c_col in cat_cols:
    for cls_idx in range(len(CLASSES)):
        train_te = np.zeros(len(train))
        test_te = np.zeros((len(test), n_splits))
        
        global_mean = (y == cls_idx).mean()
        smoothing = 10
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            
            stats = (y_tr == cls_idx).groupby(X_tr[c_col]).agg(['count', 'mean'])
            counts = stats['count']
            means = stats['mean']
            
            smooth = counts * means + smoothing * global_mean
            smooth /= (counts + smoothing)
            
            train_te[val_idx] = X_val[c_col].map(smooth).fillna(global_mean)
            test_te[:, fold] = X_test[c_col].map(smooth).fillna(global_mean)
            
        X[f'{c_col}_te_class_{cls_idx}'] = train_te
        X_test[f'{c_col}_te_class_{cls_idx}'] = test_te.mean(axis=1)

print(f"Final Enhanced Feature Count: {X.shape[1]}")

print("==========================================")
print("3. STRATIFIED 5-FOLD 5-MODEL SUPER-ENSEMBLE")
print("==========================================")

oof_lgb = np.zeros((len(train), len(CLASSES)))
oof_xgb = np.zeros((len(train), len(CLASSES)))
oof_cat = np.zeros((len(train), len(CLASSES)))
oof_et  = np.zeros((len(train), len(CLASSES)))
oof_hgb = np.zeros((len(train), len(CLASSES)))

test_lgb = np.zeros((len(test), len(CLASSES)))
test_xgb = np.zeros((len(test), len(CLASSES)))
test_cat = np.zeros((len(test), len(CLASSES)))
test_et  = np.zeros((len(test), len(CLASSES)))
test_hgb = np.zeros((len(test), len(CLASSES)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n--- Training Super-Ensemble Fold {fold + 1} / {n_splits} ---")
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # 1. LightGBM
    model_lgb = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=1500, learning_rate=0.025,
        num_leaves=63, min_child_samples=30, colsample_bytree=0.8,
        subsample=0.8, reg_alpha=1.0, reg_lambda=3.0,
        class_weight="balanced", random_state=42 + fold, n_jobs=-1, verbosity=-1
    )
    model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=120, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict_proba(X_val)
    test_lgb += model_lgb.predict_proba(X_test) / n_splits

    # 2. XGBoost
    model_xgb = xgb.XGBClassifier(
        objective="multi:softprob", num_class=len(CLASSES), n_estimators=1500,
        learning_rate=0.025, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=3.0, random_state=42 + fold, n_jobs=-1, tree_method="hist"
    )
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)
    test_xgb += model_xgb.predict_proba(X_test) / n_splits

    # 3. CatBoost
    model_cat = CatBoostClassifier(
        loss_function="MultiClass", iterations=1500, learning_rate=0.035,
        depth=6, l2_leaf_reg=3.0, auto_class_weights="Balanced",
        random_seed=42 + fold, thread_count=-1, verbose=False
    )
    model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=120, verbose=False)
    oof_cat[val_idx] = model_cat.predict_proba(X_val)
    test_cat += model_cat.predict_proba(X_test) / n_splits

    # 4. Extra Trees Classifier
    model_et = ExtraTreesClassifier(
        n_estimators=400, max_depth=30, min_samples_split=8,
        class_weight="balanced", random_state=42 + fold, n_jobs=-1
    )
    model_et.fit(X_tr, y_tr)
    oof_et[val_idx] = model_et.predict_proba(X_val)
    test_et += model_et.predict_proba(X_test) / n_splits

    # 5. Histogram Gradient Boosting Classifier
    model_hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.03, max_leaf_nodes=63,
        random_state=42 + fold, class_weight="balanced"
    )
    model_hgb.fit(X_tr, y_tr)
    oof_hgb[val_idx] = model_hgb.predict_proba(X_val)
    test_hgb += model_hgb.predict_proba(X_test) / n_splits

print("\n==========================================")
print("4. OPTIMAL SUPER-BLENDING & CALIBRATION")
print("==========================================")

oof_blend = (0.30 * oof_lgb + 
             0.25 * oof_xgb + 
             0.25 * oof_cat + 
             0.10 * oof_et  + 
             0.10 * oof_hgb)

test_blend = (0.30 * test_lgb + 
              0.25 * test_xgb + 
              0.25 * test_cat + 
              0.10 * test_et  + 
              0.10 * test_hgb)

def objective_func(weights):
    scaled_oof = oof_blend * weights
    preds = np.argmax(scaled_oof, axis=1)
    return -f1_score(y, preds, average='macro', zero_division=0)

res = minimize(objective_func, [1.0]*4, bounds=[(0.7, 1.4)]*4, method='Nelder-Mead', options={'maxiter': 500})
optimal_weights = res.x
print("Optimized Class Multipliers:", optimal_weights)

final_oof_preds = np.argmax(oof_blend * optimal_weights, axis=1)
final_macro_f1 = f1_score(y, final_oof_preds, average="macro", zero_division=0)

print(f"\n==========================================")
print(f"🚀 ELITE OOF MACRO F1 SCORE: {final_macro_f1:.5f}")
print(f"==========================================")
print("\nClassification Report:")
print(classification_report(y, final_oof_preds, target_names=CLASSES, zero_division=0))

print("==========================================")
print("5. GENERATING RANK-BEATING SUBMISSION")
print("==========================================")

final_test_preds = np.argmax(test_blend * optimal_weights, axis=1)
predicted_labels = le_target.inverse_transform(final_test_preds)

submission = pd.DataFrame({
    "id": test[ID].values,
    "Target": predicted_labels
})

assert list(submission.columns) == ["id", "Target"]
assert len(submission) == len(test)

# --- Updated output filename below ---
output_path = "/kaggle/working/network_intrusion_final_submission.csv"
submission.to_csv(output_path, index=False)

print(f"\nSubmission saved successfully as '{output_path}'!")
print("\nPrediction distribution:")
print(submission["Target"].value_counts())
print("\n🔥 Ready to submit and take Rank 2+!")