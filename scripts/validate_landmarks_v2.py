"""
Validation des predictions de landmarks par LDA -- v2.

Changements par rapport a la v1 :
  1. FIX : procrustes_alignment() contraignait implicitement une simple superposition
     orthogonale (rotation OU reflexion), sans jamais forcer det(R)=+1. Une GPA standard
     (geomorph::gpagen, MorphoJ, etc.) interdit la reflexion par defaut. Corrige ici
     (parametre allow_reflection=False par defaut, garde la possibilite de le desactiver
     pour comparaison/diagnostic).
  2. Ajout de 4 diagnostics cibles pour localiser une chute de precision anormale :
     - check_duplicate_ids       : doublons d'identifiant avant/apres le merge TPS<->CSV
     - check_group_balance       : effectifs par groupe (classes trop rares / singleton)
     - check_landmark_variance   : variance par landmark apres GPA (probleme d'homologie)
     - check_chirality           : detection de specimens en miroir AVANT alignement
  3. Validation croisee (LOOCV par defaut, ou k-fold stratifie) a la place de la
     ré-substitution pour l'accuracy rapportee -- la ré-substitution reste calculee
     et affichee a titre de comparaison / diagnostic de sur-apprentissage.
  4. Approche hierarchique : espece d'abord, puis caste au sein de l'espece, avec
     2 variantes en CV imbriquee :
       - "pipeline" : la caste est predite a partir de l'espece PREDITE (realiste)
       - "oracle"   : la caste est predite a partir de l'espece VRAIE (plafond theorique,
                      isole la difficulte "caste" de la difficulte "espece")
  5. Test de permutation optionnel (perspective #4 d'Adrien) pour juger si une accuracy
     donnee est significativement superieure au hasard.

Usage rapide :
    python validate_landmarks_lda_v2.py --tps fichier.tps --csv fichier.csv \
        --cv loo --hierarchical --permutation-test --n-permutations 200
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from pathlib import Path
import argparse

DEVICE_PATTERN = re.compile(r'_([PS]\d{1,2})(?:\.[A-Za-z0-9]+)?$')


def extract_device(value):
    if pd.isna(value):
        return None
    m = DEVICE_PATTERN.search(str(value))
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Lecture TPS + association CSV (inchange, cf. v1)
# ---------------------------------------------------------------------------

def read_tps_robust(tps_file):
    landmarks_list = []
    expected_n_lm = None
    errors = []

    with open(tps_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    current_spec_id = "unknown"

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('LM='):
            n_lm = int(line.split('=')[1])
            if expected_n_lm is None:
                expected_n_lm = n_lm
            coords = []
            i += 1

            while i < len(lines):
                coord_line = lines[i].strip()
                if coord_line.startswith('ID=') or coord_line.startswith('LM='):
                    break
                if coord_line:
                    parts = coord_line.split()
                    if len(parts) >= 2:
                        try:
                            x, y = float(parts[0]), float(parts[1])
                            coords.append([x, y])
                        except (ValueError, IndexError):
                            errors.append(f"  {current_spec_id}: ligne mal formatee: '{coord_line}'")
                i += 1

            if len(coords) != n_lm:
                errors.append(f"  {current_spec_id}: {len(coords)} landmarks lus vs {n_lm} attendus")

            if len(coords) > 0:
                landmarks_list.append({
                    'coords': np.array(coords),
                    'n_found': len(coords),
                    'spec_id': current_spec_id
                })

        elif line.startswith('ID='):
            current_spec_id = line.split('=', 1)[1].strip()
            i += 1
        else:
            i += 1

    print(f"\n{'='*70}\n📋 DIAGNOSTIC DU FICHIER TPS\n{'='*70}")
    print(f"Fichier: {tps_file}")
    print(f"Total d'entrees trouvees: {len(landmarks_list)}")
    print(f"Expected landmarks par specimen: {expected_n_lm}")

    if len(landmarks_list) == 0:
        raise ValueError("❌ Aucun landmark trouve dans le fichier TPS")

    shapes_found = {}
    for entry in landmarks_list:
        shapes_found[entry['n_found']] = shapes_found.get(entry['n_found'], 0) + 1
    print(f"\nDistribution des shapes:")
    for shape in sorted(shapes_found.keys()):
        print(f"  {shape} landmarks: {shapes_found[shape]} specimens")

    n_before = len(landmarks_list)
    landmarks_list = [s for s in landmarks_list if s['n_found'] == expected_n_lm]
    n_excluded = n_before - len(landmarks_list)
    if n_excluded > 0:
        print(f"\n🔧 Filtrage: {n_excluded} specimens exclus (≠ {expected_n_lm} landmarks)")
        print(f"   {len(landmarks_list)} specimens conserves")

    if len(landmarks_list) == 0:
        raise ValueError("❌ Aucun landmark valide apres filtrage")

    coords_array = [s['coords'] for s in landmarks_list]
    landmarks_array = np.stack(coords_array, axis=2)
    specimen_ids = [s['spec_id'] for s in landmarks_list]

    print(f"\n✅ Array final: {landmarks_array.shape}\n{'='*70}\n")
    return landmarks_array, specimen_ids


def match_tps_with_csv(landmarks, specimen_ids, df, id_col=None):
    tps_df = pd.DataFrame({'specimen_id': [str(s).strip() for s in specimen_ids],
                            'tps_index': range(len(specimen_ids))})

    candidates = [id_col] if id_col else [c for c in ['id', 'name'] if c in df.columns]
    if not candidates:
        raise ValueError(f"Aucune colonne d'ID trouvee dans le CSV parmi {list(df.columns)}. "
                          f"Precise --id-col.")

    best_col, best_merged, best_n = None, None, -1
    print(f"\n🔗 Association TPS ↔ CSV (test des colonnes candidates: {candidates})")
    for col in candidates:
        df_key = df[col].astype(str).str.strip()
        merged = tps_df.merge(
            df.assign(_match_key=df_key),
            left_on='specimen_id', right_on='_match_key', how='inner'
        )
        print(f"   colonne '{col}': {len(merged)}/{len(tps_df)} specimens TPS apparies")
        if len(merged) > best_n:
            best_col, best_merged, best_n = col, merged, len(merged)

    print(f"   → colonne retenue: '{best_col}' ({best_n} appariements)")

    if best_n == 0:
        raise ValueError(
            f"❌ Aucune correspondance trouvee entre le TPS et le CSV.\n"
            f"   Exemples d'ID TPS: {tps_df['specimen_id'].head(3).tolist()}\n"
            f"   Exemples colonne '{best_col}' du CSV: {df[best_col].astype(str).head(3).tolist()}\n"
            f"   Verifie --id-col."
        )

    n_tps_unmatched = len(tps_df) - best_n
    df_key_best = df[best_col].astype(str).str.strip()
    n_csv_unmatched = len(df) - df_key_best.isin(tps_df['specimen_id']).sum()
    if n_tps_unmatched > 0:
        print(f"   ⚠️  {n_tps_unmatched} specimens TPS sans correspondance CSV (exclus)")
    if n_csv_unmatched > 0:
        print(f"   ⚠️  {n_csv_unmatched} lignes CSV sans correspondance TPS (exclues)")

    merged = best_merged.drop(columns=['_match_key']).reset_index(drop=True)
    landmarks_matched = landmarks[:, :, merged['tps_index'].values]

    return landmarks_matched, merged, best_col


# ---------------------------------------------------------------------------
# NOUVEAU : diagnostics cibles
# ---------------------------------------------------------------------------

def check_duplicate_ids(specimen_ids, df, matched_col):
    """Doublons d'ID : la cause la plus courante et la plus silencieuse de
    desynchronisation landmarks <-> etiquette apres un merge (many-to-one)."""
    print(f"\n{'='*70}\n🔍 DOUBLONS D'IDENTIFIANT\n{'='*70}")

    tps_dupes = pd.Series(specimen_ids).value_counts()
    tps_dupes = tps_dupes[tps_dupes > 1]
    if len(tps_dupes) > 0:
        print(f"⚠️  {len(tps_dupes)} identifiant(s) TPS duplique(s) AVANT matching :")
        print(tps_dupes.to_string())
    else:
        print("✓ Aucun identifiant TPS duplique avant matching.")

    csv_dupes = df[matched_col].astype(str).str.strip().value_counts()
    csv_dupes = csv_dupes[csv_dupes > 1]
    if len(csv_dupes) > 0:
        print(f"\n⚠️  {len(csv_dupes)} identifiant(s) '{matched_col}' present(s) plusieurs fois "
              f"APRES matching :")
        print(csv_dupes.to_string())
        print("   -> Chaque doublon correspond a un merge many-to-one : le meme jeu de landmarks")
        print("      se retrouve associe a plusieurs lignes de metadonnees (ou l'inverse), ce qui")
        print("      decale silencieusement la correspondance forme <-> etiquette pour tout le reste")
        print("      du dataframe si l'ordre n'est plus 1-a-1.")
    else:
        print("✓ Aucun identifiant duplique apres matching : correspondance 1-a-1 preservee.")
    print(f"{'='*70}\n")
    return tps_dupes, csv_dupes


def check_group_balance(df, group_col='groupe', min_n=5):
    print(f"\n{'='*70}\n🔍 EQUILIBRE DES GROUPES ('{group_col}')\n{'='*70}")
    counts = df[group_col].value_counts()
    print(counts.to_string())

    singleton = counts[counts < 2]
    rare = counts[(counts >= 2) & (counts < min_n)]
    if len(singleton) > 0:
        print(f"\n❌ {len(singleton)} groupe(s) avec 1 seul individu : "
              f"impossibles a valider en LOOCV (0% garanti pour ce groupe, aucun exemple "
              f"d'entrainement ne reste une fois l'individu retire) :")
        print(singleton.to_string())
    if len(rare) > 0:
        print(f"\n⚠️  {len(rare)} groupe(s) avec moins de {min_n} individus (LDA/CV peu fiable) :")
        print(rare.to_string())
    if len(singleton) == 0 and len(rare) == 0:
        print(f"\n✓ Tous les groupes ont au moins {min_n} individus.")
    print(f"{'='*70}\n")
    return counts


def check_landmark_variance(landmarks_aligned):
    """Variance par landmark apres GPA : un point avec une variance nettement superieure
    aux autres suggere un probleme d'homologie (ordre incoherent entre certains specimens,
    point pas toujours pose au meme endroit anatomique)."""
    print(f"\n{'='*70}\n🔍 VARIANCE PAR LANDMARK APRES GPA\n{'='*70}")
    consensus = landmarks_aligned.mean(axis=2, keepdims=True)
    diffs = landmarks_aligned - consensus
    var_per_lm = (diffs ** 2).sum(axis=1).mean(axis=1)

    mean_v, std_v = var_per_lm.mean(), var_per_lm.std()
    for idx, v in enumerate(var_per_lm):
        flag = "  ⚠️ variance anormalement elevee" if v > mean_v + 2 * std_v else ""
        print(f"  landmark {idx:2d}: variance = {v:.6f}{flag}")

    ratio = var_per_lm.max() / (var_per_lm.min() + 1e-12)
    print(f"\nRatio variance max/min inter-landmark: {ratio:.1f}")
    if ratio > 20:
        print("⚠️  Un ecart aussi fort entre landmarks suggere souvent une incoherence")
        print("   d'homologie plutot qu'une simple variabilite biologique.")
    print(f"{'='*70}\n")
    return var_per_lm


def check_chirality(landmarks):
    """Detecte des specimens en miroir AVANT alignement, via l'aire signee du polygone
    forme par les landmarks (formule du lacet). Si les landmarks sont dans le meme ordre
    anatomique pour tous les specimens, l'aire signee doit avoir un signe constant.
    Un melange de signes = specimens digitalises avec une chiralite differente
    (ex: aile gauche/droite non re-normalisee, ou image parfois retournee)."""
    print(f"\n{'='*70}\n🔍 CHIRALITE (aire signee des landmarks bruts, avant GPA)\n{'='*70}")
    n_spec = landmarks.shape[2]
    signed_areas = np.zeros(n_spec)
    for i in range(n_spec):
        pts = landmarks[:, :, i]
        x, y = pts[:, 0], pts[:, 1]
        signed_areas[i] = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

    n_pos = (signed_areas > 0).sum()
    n_neg = (signed_areas < 0).sum()
    minority = min(n_pos, n_neg)
    print(f"Specimens avec aire signee positive: {n_pos}")
    print(f"Specimens avec aire signee negative: {n_neg}")
    if minority > 0 and minority / n_spec > 0.02:
        print(f"⚠️  {minority}/{n_spec} specimens ({100*minority/n_spec:.1f}%) ont une chiralite "
              f"opposee a la majorite.")
        print("   -> Ce sont des candidats a une inversion miroir (mauvais wing side, image")
        print("      retournee, ou ordre de digitalisation inverse). A verifier specimen par")
        print("      specimen avant de les re-mirorer ou de les exclure.")
    else:
        print("✓ Chiralite homogene sur l'ensemble du dataset.")
    print(f"{'='*70}\n")
    return signed_areas


# ---------------------------------------------------------------------------
# GPA -- FIX : contrainte det(R) = +1 (pas de reflexion) par defaut
# ---------------------------------------------------------------------------

def procrustes_alignment(landmarks, allow_reflection=False, max_iter=30, tol=1e-6):
    """GPA iterative (SVD). Par defaut, interdit la reflexion (comme geomorph::gpagen,
    MorphoJ, etc.) en forcant det(R)=+1 -- la v1 ne le faisait pas, ce qui autorisait
    silencieusement des reflexions lors de la superposition."""
    n_lm, n_dims, n_spec = landmarks.shape

    centered = landmarks - landmarks.mean(axis=0, keepdims=True)
    centroid_size = np.sqrt((centered ** 2).sum(axis=(0, 1)))
    scaled = centered / centroid_size[np.newaxis, np.newaxis, :]

    consensus = scaled.mean(axis=2, keepdims=True)
    converged = False

    for iteration in range(max_iter):
        aligned = np.zeros_like(scaled)
        for i in range(n_spec):
            H = scaled[:, :, i].T @ consensus[:, :, 0]
            U, S, Vt = svd(H)
            if allow_reflection:
                R = U @ Vt
            else:
                d = np.sign(np.linalg.det(U @ Vt))
                D = np.eye(n_dims)
                D[-1, -1] = d
                R = U @ D @ Vt
            aligned[:, :, i] = scaled[:, :, i] @ R.T

        consensus_old = consensus.copy()
        consensus = aligned.mean(axis=2, keepdims=True)

        if np.allclose(consensus, consensus_old, atol=tol):
            print(f"  Convergence GPA atteinte a l'iteration {iteration+1}")
            converged = True
            break

    if not converged:
        print(f"  ⚠️  GPA non convergee apres {max_iter} iterations (tol={tol}). "
              f"Resultat utilise quand meme, mais a verifier.")

    return aligned


def diagnostic_outliers(landmarks_aligned, specimen_ids, output_dir, threshold_std=2.0):
    consensus = landmarks_aligned.mean(axis=2, keepdims=True)
    diffs = landmarks_aligned - consensus
    proc_dist = np.sqrt((diffs ** 2).sum(axis=(0, 1)))

    mean_d, std_d = proc_dist.mean(), proc_dist.std()
    threshold = mean_d + threshold_std * std_d
    outlier_mask = proc_dist > threshold
    n_outliers = outlier_mask.sum()

    print(f"\n{'='*70}\n🔍 DIAGNOSTIC OUTLIERS (distance Procrustes au consensus)\n{'='*70}")
    print(f"Distance moyenne: {mean_d:.4f} (std: {std_d:.4f})")
    print(f"Seuil (moyenne + {threshold_std}*std): {threshold:.4f}")
    print(f"Specimens suspects: {n_outliers}/{len(proc_dist)} ({100*n_outliers/len(proc_dist):.1f}%)")

    if n_outliers > 0:
        print(f"\nTop 10 specimens les plus suspects:")
        idx_sorted = np.argsort(proc_dist)[::-1][:10]
        for idx in idx_sorted:
            print(f"  {specimen_ids[idx]}: distance = {proc_dist[idx]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(proc_dist, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(threshold, color='red', linestyle='--', label=f'Seuil ({threshold_std}σ)')
    ax.set_xlabel('Distance Procrustes au consensus')
    ax.set_ylabel('Nombre de specimens')
    ax.set_title('Distribution des distances Procrustes')
    ax.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'procrustes_distance_diagnostic.png', dpi=150)
    plt.close(fig)
    print(f"\n✓ Histogramme sauvegarde: {Path(output_dir) / 'procrustes_distance_diagnostic.png'}")
    print(f"{'='*70}\n")

    return proc_dist, outlier_mask


# ---------------------------------------------------------------------------
# NOUVEAU : validation croisee (remplace la ré-substitution pour le chiffre reporte)
# ---------------------------------------------------------------------------

def get_cv_splitter(y, cv='loo', k=5, random_state=0):
    if cv == 'loo':
        return LeaveOneOut()
    elif cv == 'kfold':
        return StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    else:
        raise ValueError("cv doit etre 'loo' ou 'kfold'")


def run_flat_lda(X_pca, y, cv='loo', k=5, random_state=0):
    """LDA a plat (toutes les classes groupe=espece_caste), avec ré-substitution
    (diagnostic de sur-apprentissage) ET validation croisee (chiffre honnete a reporter,
    equivalent du CV=TRUE de MASS::lda utilise par Adrien)."""
    y = np.asarray(y)

    lda_full = LinearDiscriminantAnalysis()
    lda_full.fit(X_pca, y)
    y_pred_resub = lda_full.predict(X_pca)
    acc_resub = np.mean(y == y_pred_resub)

    splitter = get_cv_splitter(y, cv=cv, k=k, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred_cv = cross_val_predict(LinearDiscriminantAnalysis(), X_pca, y, cv=splitter)
    acc_cv = np.mean(y == y_pred_cv)

    return {
        'accuracy_resubstitution': acc_resub,
        'accuracy_cv': acc_cv,
        'y_pred_resubstitution': y_pred_resub,
        'y_pred_cv': y_pred_cv,
        'lda_full_for_plot': lda_full,  # pour la projection 2D uniquement (fit sur tout, illustratif)
    }


# ---------------------------------------------------------------------------
# NOUVEAU : approche hierarchique (espece -> caste), en CV imbriquee
# ---------------------------------------------------------------------------

def _fit_predict_caste(X_pca, y_caste, train_idx, test_idx):
    """Ajuste un LDA caste sur train_idx et predit sur test_idx (1 seul point ici,
    mais generalise a un batch). Repli sur la classe majoritaire si trop peu de donnees
    (classe absente, ou < 2 exemplaires dans une classe -> LDA impossible/instable)."""
    sub_y = y_caste[train_idx]
    classes, counts = np.unique(sub_y, return_counts=True)

    if len(classes) == 0:
        return np.array(['unknown'] * len(test_idx), dtype=object)
    if len(classes) == 1 or counts.min() < 2 or len(train_idx) <= len(classes):
        majority = classes[np.argmax(counts)]
        return np.array([majority] * len(test_idx), dtype=object)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_pca[train_idx], sub_y)
    return lda.predict(X_pca[test_idx])


def hierarchical_cv(X_pca, df, cv='loo', k=5, random_state=0,
                     espece_col='espece', caste_col='caste'):
    """CV imbriquee a 2 etages :
       1) espece predite (LOOCV ou k-fold stratifie)
       2) caste predite, en 2 variantes :
          - 'pipeline' : le modele caste utilise pour un individu est celui entraine sur les
                         AUTRES individus dont l'espece PREDITE (au point 1) correspond a
                         l'espece predite pour cet individu -- c'est le comportement reel
                         d'un pipeline en deploiement (on ne connait jamais la vraie espece).
          - 'oracle'   : meme chose mais avec la VRAIE espece -- plafond theorique qui isole
                         la difficulte de la discrimination caste de celle de l'espece.
    """
    y_espece = df[espece_col].values
    y_caste = df[caste_col].values
    n = len(df)

    splitter = get_cv_splitter(y_espece, cv=cv, k=k, random_state=random_state)

    espece_pred = np.empty(n, dtype=object)
    caste_pred_pipeline = np.empty(n, dtype=object)
    caste_pred_oracle = np.empty(n, dtype=object)

    for train_idx, test_idx in splitter.split(X_pca, y_espece):
        lda_esp = LinearDiscriminantAnalysis()
        lda_esp.fit(X_pca[train_idx], y_espece[train_idx])
        pred_esp_batch = lda_esp.predict(X_pca[test_idx])
        espece_pred[test_idx] = pred_esp_batch

        for local_i, i in enumerate(test_idx):
            esp_hat = pred_esp_batch[local_i]
            esp_true = y_espece[i]

            sub_train_pred = train_idx[y_espece[train_idx] == esp_hat]
            caste_pred_pipeline[i] = _fit_predict_caste(
                X_pca, y_caste, sub_train_pred, np.array([i]))[0]

            sub_train_true = train_idx[y_espece[train_idx] == esp_true]
            caste_pred_oracle[i] = _fit_predict_caste(
                X_pca, y_caste, sub_train_true, np.array([i]))[0]

    groupe_pred_pipeline = np.array([f"{e}_{c}" for e, c in zip(espece_pred, caste_pred_pipeline)])
    groupe_pred_oracle = np.array([f"{e}_{c}" for e, c in zip(y_espece, caste_pred_oracle)])
    groupe_true = np.array([f"{e}_{c}" for e, c in zip(y_espece, y_caste)])

    return {
        'espece_pred': espece_pred,
        'accuracy_espece': np.mean(y_espece == espece_pred),
        'caste_pred_pipeline': caste_pred_pipeline,
        'accuracy_groupe_pipeline': np.mean(groupe_true == groupe_pred_pipeline),
        'caste_pred_oracle': caste_pred_oracle,
        'accuracy_caste_given_true_espece': np.mean(y_caste == caste_pred_oracle),
        'accuracy_groupe_oracle': np.mean(groupe_true == groupe_pred_oracle),
        'groupe_pred_pipeline': groupe_pred_pipeline,
        'groupe_pred_oracle': groupe_pred_oracle,
    }


# ---------------------------------------------------------------------------
# NOUVEAU : test de permutation (perspective #4 d'Adrien)
# ---------------------------------------------------------------------------

def permutation_test(X_pca, y, n_permutations=200, cv='kfold', k=5, random_state=0):
    """Compare l'accuracy CV observee a une distribution nulle obtenue en permutant les
    etiquettes. Utilise k-fold par defaut (plus rapide que LOOCV pour N permutations).
    Avec 30 classes, le hasard pur donne ~1/30 = 3.3% d'accuracy attendue."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)

    splitter = get_cv_splitter(y, cv=cv, k=k, random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred_obs = cross_val_predict(LinearDiscriminantAnalysis(), X_pca, y, cv=splitter)
    obs_acc = np.mean(y == y_pred_obs)

    null_accs = np.zeros(n_permutations)
    for p in range(n_permutations):
        y_perm = rng.permutation(y)
        splitter_p = get_cv_splitter(y_perm, cv=cv, k=k, random_state=random_state)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_pred_p = cross_val_predict(LinearDiscriminantAnalysis(), X_pca, y_perm, cv=splitter_p)
            null_accs[p] = np.mean(y_perm == y_pred_p)
        except ValueError:
            null_accs[p] = np.nan

    null_accs = null_accs[~np.isnan(null_accs)]
    p_value = (np.sum(null_accs >= obs_acc) + 1) / (len(null_accs) + 1)

    print(f"\n{'='*70}\n🎲 TEST DE PERMUTATION\n{'='*70}")
    print(f"Accuracy observee (CV): {100*obs_acc:.2f}%")
    print(f"Accuracy nulle moyenne (labels permutes, n={len(null_accs)}): "
          f"{100*null_accs.mean():.2f}% (std {100*null_accs.std():.2f}%)")
    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("-> L'accuracy observee est significativement superieure au hasard.")
    else:
        print("-> L'accuracy observee N'EST PAS distinguable du hasard a 5%.")
    print(f"{'='*70}\n")

    return {'observed_accuracy': obs_acc, 'null_distribution': null_accs, 'p_value': p_value}


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def validate_landmarks(tps_file, csv_file, output_dir="./validation_results",
                        id_col=None, device=None, device_col=None,
                        cv='loo', kfold_k=5, allow_reflection=False,
                        hierarchical=False, run_permutation_test=False,
                        n_permutations=200, espece_col='espece', caste_col='caste'):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*70 + "\nVALIDATION DES LANDMARKS PAR LDA (v2)\n" + "="*70)

    print(f"\n[1/8] Lecture TPS: {tps_file}")
    landmarks, specimen_ids = read_tps_robust(tps_file)
    print(f"  ✓ {landmarks.shape[2]} specimens, {landmarks.shape[0]} landmarks")

    print(f"[2/8] Lecture CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  ✓ {len(df)} lignes, colonnes: {list(df.columns)}")

    print(f"[3/8] Association TPS ↔ CSV par identifiant")
    landmarks, df, matched_col = match_tps_with_csv(landmarks, specimen_ids, df, id_col=id_col)
    specimen_ids = df['specimen_id'].tolist()
    print(f"  ✓ {landmarks.shape[2]} specimens apres association (colonne '{matched_col}')")

    # --- diagnostic doublons : juste apres le merge, avant tout calcul ---
    check_duplicate_ids(specimen_ids, df, matched_col)

    print(f"[3bis/8] Filtrage par appareil")
    if device_col and device_col in df.columns:
        df['device'] = df[device_col]
    elif 'device' in df.columns:
        pass
    else:
        df['device'] = df['specimen_id'].apply(extract_device)
        n_parsed = df['device'].notna().sum()
        print(f"  ✓ Device parse depuis '{matched_col}': {n_parsed}/{len(df)} reconnus")

    if device is not None:
        n_before = len(df)
        keep_mask = df['device'] == device
        df = df[keep_mask].reset_index(drop=True)
        landmarks = landmarks[:, :, keep_mask.values]
        print(f"  🔧 Filtre --device={device}: {n_before} → {len(df)} specimens")
        if len(df) == 0:
            raise ValueError(f"❌ Aucun specimen ne correspond a l'appareil '{device}'.")
    else:
        print(f"  (pas de filtre --device : tous les appareils sont conserves)")

    print(f"[4/8] Creation variable Groupe")
    if caste_col not in df.columns:
        raise ValueError(f"Colonne '{caste_col}' absente du CSV. Colonnes disponibles: {list(df.columns)}")
    n_missing_caste = df[caste_col].isna().sum()
    if n_missing_caste > 0:
        print(f"  ⚠️  {n_missing_caste} valeur(s) manquante(s) dans '{caste_col}' "
              f"(vont creer un groupe '..._nan' artificiel)")
    df['groupe'] = df[espece_col].astype(str) + "_" + df[caste_col].astype(str)
    print(f"  ✓ {df['groupe'].nunique()} groupes")

    check_group_balance(df, group_col='groupe', min_n=5)

    print(f"[5/8] Chiralite (avant GPA) + Alignement Procrustes "
          f"(reflexion {'autorisee' if allow_reflection else 'interdite (standard GM)'})")
    check_chirality(landmarks)
    landmarks_aligned = procrustes_alignment(landmarks, allow_reflection=allow_reflection)

    proc_dist, outlier_mask = diagnostic_outliers(landmarks_aligned, specimen_ids, output_dir)
    df['procrustes_distance'] = proc_dist
    df['is_outlier'] = outlier_mask

    check_landmark_variance(landmarks_aligned)

    device_outlier_stats = df.groupby('device', dropna=False)['is_outlier'].agg(['sum', 'count'])
    device_outlier_stats['pct_outlier'] = 100 * device_outlier_stats['sum'] / device_outlier_stats['count']
    device_outlier_stats = device_outlier_stats.rename(columns={'sum': 'n_outliers', 'count': 'n_total'})
    device_outlier_stats.to_csv(Path(output_dir) / 'outliers_by_device.csv')

    print(f"[6/8] Analyse PCA")
    X_flat = landmarks_aligned.reshape(landmarks_aligned.shape[0] * landmarks_aligned.shape[1], -1).T
    pca = PCA()
    X_pca_full = pca.fit_transform(X_flat)
    n_comp = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.95)[0][0] + 1
    X_pca = X_pca_full[:, :n_comp]
    var_exp = 100 * np.sum(pca.explained_variance_ratio_[:n_comp])
    print(f"  ✓ {n_comp} composantes ({var_exp:.1f}% variance)")
    print(f"  (Note: la PCA est calculee une seule fois sur tout le jeu de donnees, comme dans")
    print(f"   le script R d'Adrien -- seule l'etape LDA est cross-validee. Une PCA re-ajustee")
    print(f"   a chaque fold serait plus stricte mais casserait la comparaison directe avec")
    print(f"   son chiffre de reference.)")

    print(f"[7/8] LDA a plat (30 groupes espece×caste), cv='{cv}'")
    flat_results = run_flat_lda(X_pca, df['groupe'].values, cv=cv, k=kfold_k)
    print(f"  Ré-substitution (optimiste, diagnostic de sur-apprentissage): "
          f"{100*flat_results['accuracy_resubstitution']:.2f}%")
    print(f"  Validation croisee ('{cv}', chiffre a reporter) : "
          f"{100*flat_results['accuracy_cv']:.2f}%")

    cm = confusion_matrix(df['groupe'], flat_results['y_pred_cv'], labels=sorted(df['groupe'].unique()))
    cm_df = pd.DataFrame(cm, index=sorted(df['groupe'].unique()), columns=sorted(df['groupe'].unique()))
    print(f"\nRapport par groupe (CV):\n"
          f"{classification_report(df['groupe'], flat_results['y_pred_cv'], digits=3, zero_division=0)}")

    hier_results = None
    if hierarchical:
        print(f"\n[7bis/8] Approche hierarchique (espece -> caste), cv='{cv}'")
        hier_results = hierarchical_cv(X_pca, df, cv=cv, k=kfold_k,
                                        espece_col=espece_col, caste_col=caste_col)
        print(f"  Accuracy espece (etage 1) ................................ "
              f"{100*hier_results['accuracy_espece']:.2f}%")
        print(f"  Accuracy caste | espece VRAIE (oracle, plafond) ........... "
              f"{100*hier_results['accuracy_caste_given_true_espece']:.2f}%")
        print(f"  Accuracy groupe complet | pipeline realiste (espece predite) "
              f"{100*hier_results['accuracy_groupe_pipeline']:.2f}%")
        print(f"  Accuracy groupe complet | oracle espece ................... "
              f"{100*hier_results['accuracy_groupe_oracle']:.2f}%")
        print(f"  Comparer ceci a l'accuracy a plat ({100*flat_results['accuracy_cv']:.2f}%) :")
        print(f"  si l'oracle est nettement meilleur que le pipeline, l'espece est le goulot")
        print(f"  d'etranglement ; si les deux sont proches de l'accuracy a plat, la structure")
        print(f"  hierarchique n'apporte pas grand-chose et le probleme est plus profond (donnees).")

    perm_results = None
    if run_permutation_test:
        print(f"\n[8/8] Test de permutation (n={n_permutations})")
        perm_results = permutation_test(X_pca, df['groupe'].values, n_permutations=n_permutations, cv='kfold', k=kfold_k)

    # --- graphiques (projection LDA illustrative, fit sur tout le dataset) ---
    print(f"\nGeneration des graphiques...")
    groups = sorted(df['groupe'].unique())
    n_groups = len(groups)
    fig, axes = plt.subplots(1, 2, figsize=(max(14, n_groups * 0.6), max(5, n_groups * 0.5)))

    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=True,
                xticklabels=True, yticklabels=True, annot_kws={"size": 7})
    axes[0].set_title('Matrice de confusion (LDA, predictions CV)')
    axes[0].set_ylabel('Groupe reel')
    axes[0].set_xlabel('Groupe predit (CV)')
    axes[0].tick_params(axis='x', rotation=90, labelsize=8)
    axes[0].tick_params(axis='y', rotation=0, labelsize=8)

    lda_full = flat_results['lda_full_for_plot']
    lda_transform = lda_full.transform(X_pca)
    cmap_combined = np.vstack([plt.cm.tab20(np.linspace(0, 1, 20)), plt.cm.tab20b(np.linspace(0, 1, 20))])
    colors = cmap_combined[:len(groups)]

    for i, group in enumerate(groups):
        mask = (df['groupe'] == group).values
        y_coord = lda_transform[mask, 1] if lda_transform.shape[1] > 1 else np.zeros(mask.sum())
        axes[1].scatter(lda_transform[mask, 0], y_coord, label=group, alpha=0.6, s=50, color=colors[i])

    axes[1].set_xlabel(f"LD1 ({100*lda_full.explained_variance_ratio_[0]:.1f}%)")
    if lda_transform.shape[1] > 1:
        axes[1].set_ylabel(f"LD2 ({100*lda_full.explained_variance_ratio_[1]:.1f}%)")
    axes[1].set_title('Projection LDA (fit sur tout le dataset -- illustratif, PAS le chiffre CV)')
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'lda_validation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    cm_df.to_csv(output_dir / 'confusion_matrix_cv.csv')
    df['Prediction_CV'] = flat_results['y_pred_cv']
    df['Prediction_resubstitution'] = flat_results['y_pred_resubstitution']
    df['Correct_CV'] = (df['groupe'] == df['Prediction_CV']).astype(int)
    if hier_results is not None:
        df['espece_pred_CV'] = hier_results['espece_pred']
        df['caste_pred_pipeline_CV'] = hier_results['caste_pred_pipeline']
        df['caste_pred_oracle_CV'] = hier_results['caste_pred_oracle']
    df.to_csv(output_dir / 'predictions_detailed.csv', index=False)

    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("VALIDATION DES LANDMARKS PAR LDA (v2)\n" + "="*70 + "\n\n")
        f.write(f"Fichier TPS: {tps_file}\nFichier CSV: {csv_file}\n")
        f.write(f"Reflexion autorisee dans la GPA: {allow_reflection}\n")
        f.write(f"Specimens: {landmarks.shape[2]}, Landmarks: {landmarks.shape[0]}\n")
        f.write(f"Groupes: {df['groupe'].nunique()}\n\n")
        f.write(f"PCA: {n_comp} composantes ({var_exp:.1f}% variance)\n\n")
        f.write(f"ACCURACY RE-SUBSTITUTION: {100*flat_results['accuracy_resubstitution']:.2f}%\n")
        f.write(f"ACCURACY CV ('{cv}'): {100*flat_results['accuracy_cv']:.2f}%\n\n")
        if hier_results is not None:
            f.write(f"ACCURACY ESPECE (etage 1): {100*hier_results['accuracy_espece']:.2f}%\n")
            f.write(f"ACCURACY CASTE | ESPECE VRAIE (oracle): {100*hier_results['accuracy_caste_given_true_espece']:.2f}%\n")
            f.write(f"ACCURACY GROUPE | PIPELINE REALISTE: {100*hier_results['accuracy_groupe_pipeline']:.2f}%\n")
            f.write(f"ACCURACY GROUPE | ORACLE ESPECE: {100*hier_results['accuracy_groupe_oracle']:.2f}%\n\n")
        if perm_results is not None:
            f.write(f"TEST DE PERMUTATION: accuracy obs={100*perm_results['observed_accuracy']:.2f}%, "
                    f"p-value={perm_results['p_value']:.4f}\n\n")
        f.write(cm_df.to_string())

    print(f"\n{'='*70}\n✅ Validation terminee!\n{'='*70}\n")

    return {
        'flat': flat_results,
        'hierarchical': hier_results,
        'permutation': perm_results,
        'confusion_matrix': cm_df,
        'data': df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validation des landmarks par LDA (v2)')
    parser.add_argument('--tps', required=True)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', default='./out/validation_results')
    parser.add_argument('--id-col', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--device-col', default=None)
    parser.add_argument('--cv', choices=['loo', 'kfold'], default='loo',
                         help="Methode de validation croisee pour le chiffre reporte (defaut: loo, "
                              "equivalent du CV=TRUE d'Adrien en R)")
    parser.add_argument('--kfold-k', type=int, default=5, help="Nombre de folds si --cv kfold")
    parser.add_argument('--allow-reflection', action='store_true',
                         help="Autorise la reflexion dans la GPA (deconseille -- pour diagnostic uniquement)")
    parser.add_argument('--hierarchical', action='store_true',
                         help="Ajoute l'evaluation hierarchique espece -> caste")
    parser.add_argument('--permutation-test', action='store_true',
                         help="Ajoute un test de permutation sur l'accuracy a plat")
    parser.add_argument('--n-permutations', type=int, default=200)
    parser.add_argument('--espece-col', default='espece')
    parser.add_argument('--caste-col', default='caste')

    args = parser.parse_args()

    validate_landmarks(
        args.tps, args.csv, args.out,
        id_col=args.id_col, device=args.device, device_col=args.device_col,
        cv=args.cv, kfold_k=args.kfold_k, allow_reflection=args.allow_reflection,
        hierarchical=args.hierarchical, run_permutation_test=args.permutation_test,
        n_permutations=args.n_permutations, espece_col=args.espece_col, caste_col=args.caste_col,
    )