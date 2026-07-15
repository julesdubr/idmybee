"""
Validation des predictions de landmarks par LDA.
- Filtre robuste TPS (18 landmarks) + merge par ID avec le CSV (au lieu de troncature positionnelle)
- Filtrage optionnel par appareil photo (device), parse depuis le nom ou colonne CSV dediee
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from pathlib import Path
import argparse

# Pattern par defaut pour extraire l'appareil depuis un nom de fichier: '..._P3' ou '..._S12' (avec extension optionnelle)
DEVICE_PATTERN = re.compile(r'_([PS]\d{1,2})(?:\.[A-Za-z0-9]+)?$')


def extract_device(value):
    """Extrait le code appareil (ex: 'P3', 'S12') depuis un nom de fichier/ID."""
    if pd.isna(value):
        return None
    m = DEVICE_PATTERN.search(str(value))
    return m.group(1) if m else None


def read_tps_robust(tps_file):
    """
    Lit un fichier TPS avec diagnostic de robustesse.
    Ne garde que les specimens ayant EXACTEMENT le nombre de landmarks attendu (18).
    """
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

    print(f"\n{'='*70}")
    print(f"📋 DIAGNOSTIC DU FICHIER TPS")
    print(f"{'='*70}")
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

    print(f"\n✅ Array final: {landmarks_array.shape}")
    print(f"{'='*70}\n")

    return landmarks_array, specimen_ids


def match_tps_with_csv(landmarks, specimen_ids, df, id_col=None):
    """
    Associe les landmarks TPS aux metadonnees CSV par ID (pas par position).
    Le TPS ID= est lu comme string ; le CSV 'id' peut etre int -> on caste les deux
    en string pour la comparaison, pour eviter un mismatch de type silencieux.
    Si id_col n'est pas fourni, essaie 'id' puis 'name' et garde celle qui matche le mieux.
    Retourne les landmarks reordonnes/filtres + le dataframe fusionne (meme ordre).
    """
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


def procrustes_alignment(landmarks):
    """Alignement Procrustes generalise (GPA) avec rotation 2D."""
    n_lm, n_dims, n_spec = landmarks.shape

    centered = landmarks - landmarks.mean(axis=0, keepdims=True)
    centroid_size = np.sqrt((centered ** 2).sum(axis=(0, 1)))
    scaled = centered / centroid_size[np.newaxis, np.newaxis, :]

    consensus = scaled.mean(axis=2, keepdims=True)

    for iteration in range(10):
        aligned = np.zeros_like(scaled)
        for i in range(n_spec):
            H = scaled[:, :, i].T @ consensus[:, :, 0]
            U, _, Vt = svd(H)
            R = U @ Vt
            aligned[:, :, i] = scaled[:, :, i] @ R.T

        consensus_old = consensus.copy()
        consensus = aligned.mean(axis=2, keepdims=True)

        if np.allclose(consensus, consensus_old, atol=1e-6):
            print(f"  Convergence atteinte a l'iteration {iteration+1}")
            break

    return aligned


def diagnostic_outliers(landmarks_aligned, specimen_ids, output_dir, threshold_std=2.0):
    """Detecte les specimens mal alignes (distance Procrustes elevee au consensus)."""
    consensus = landmarks_aligned.mean(axis=2, keepdims=True)
    diffs = landmarks_aligned - consensus
    proc_dist = np.sqrt((diffs ** 2).sum(axis=(0, 1)))

    mean_d, std_d = proc_dist.mean(), proc_dist.std()
    threshold = mean_d + threshold_std * std_d
    outlier_mask = proc_dist > threshold
    n_outliers = outlier_mask.sum()

    print(f"\n{'='*70}")
    print(f"🔍 DIAGNOSTIC OUTLIERS (distance Procrustes au consensus)")
    print(f"{'='*70}")
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
    print(f"\n✓ Histogramme sauvegarde: {Path(output_dir) / 'procrustes_distance_diagnostic.png'}")
    print(f"{'='*70}\n")

    return proc_dist, outlier_mask


def validate_landmarks(tps_file, csv_file, output_dir="./validation_results",
                        id_col=None, device=None, device_col=None):
    """Pipeline complet de validation par LDA."""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*70)
    print("VALIDATION DES LANDMARKS PAR LDA")
    print("="*70)

    # 1. Lire TPS (deja filtre a 18 landmarks)
    print(f"\n[1/7] Lecture TPS: {tps_file}")
    landmarks, specimen_ids = read_tps_robust(tps_file)
    print(f"  ✓ {landmarks.shape[2]} specimens, {landmarks.shape[0]} landmarks")

    # 2. Lire CSV
    print(f"[2/7] Lecture CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  ✓ {len(df)} lignes, colonnes: {list(df.columns)}")

    # 3. Associer TPS <-> CSV par ID (pas par position)
    print(f"[3/7] Association TPS ↔ CSV par identifiant")
    landmarks, df, matched_col = match_tps_with_csv(landmarks, specimen_ids, df, id_col=id_col)
    specimen_ids = df['specimen_id'].tolist()
    print(f"  ✓ {landmarks.shape[2]} specimens apres association (colonne '{matched_col}')")

    # 3bis. Filtrage par appareil
    print(f"[3bis/7] Filtrage par appareil")
    if device_col and device_col in df.columns:
        df['device'] = df[device_col]
        print(f"  ✓ Colonne '{device_col}' utilisee directement comme device")
    elif 'device' in df.columns:
        print(f"  ✓ Colonne 'device' deja presente dans le CSV, utilisee telle quelle")
    else:
        df['device'] = df['specimen_id'].apply(extract_device)
        n_parsed = df['device'].notna().sum()
        print(f"  ✓ Device parse depuis '{matched_col}' (regex '_P#'/'_S#'): {n_parsed}/{len(df)} reconnus")

    print(f"  Distribution des appareils:\n{df['device'].value_counts(dropna=False).to_string()}")

    if device is not None:
        n_before = len(df)
        keep_mask = df['device'] == device
        df = df[keep_mask].reset_index(drop=True)
        landmarks = landmarks[:, :, keep_mask.values]
        print(f"  🔧 Filtre --device={device}: {n_before} → {len(df)} specimens")
        if len(df) == 0:
            raise ValueError(f"❌ Aucun specimen ne correspond a l'appareil '{device}'. "
                              f"Valeurs disponibles: {sorted(df['device'].dropna().unique()) if len(df) else 'aucune'}")
    else:
        print(f"  (pas de filtre --device : tous les appareils sont conserves)")

    # 4. Creer Groupe
    print(f"[4/7] Creation variable Groupe")
    df['groupe'] = df['espece'].astype(str) + "_" + df['role'].astype(str)
    print(f"  ✓ {df['groupe'].nunique()} groupes")

    # 5. Procrustes
    print(f"[5/7] Alignement Procrustes")
    landmarks_aligned = procrustes_alignment(landmarks)
    print(f"  ✓ Alignement termine")

    proc_dist, outlier_mask = diagnostic_outliers(landmarks_aligned, specimen_ids, output_dir)
    df['procrustes_distance'] = proc_dist
    df['is_outlier'] = outlier_mask

    # Croisement outlier x device : un appareil concentre-t-il les mauvais alignements ?
    print(f"\n{'='*70}")
    print(f"🔍 CROISEMENT OUTLIERS × APPAREIL")
    print(f"{'='*70}")
    device_outlier_stats = df.groupby('device', dropna=False)['is_outlier'].agg(['sum', 'count'])
    device_outlier_stats['pct_outlier'] = 100 * device_outlier_stats['sum'] / device_outlier_stats['count']
    device_outlier_stats = device_outlier_stats.rename(columns={'sum': 'n_outliers', 'count': 'n_total'})
    device_outlier_stats = device_outlier_stats.sort_values('pct_outlier', ascending=False)
    print(device_outlier_stats.to_string())
    device_outlier_stats.to_csv(Path(output_dir) / 'outliers_by_device.csv')
    print(f"\n✓ Sauvegarde: {Path(output_dir) / 'outliers_by_device.csv'}")
    print(f"  (si un appareil a un % d'outliers nettement plus eleve que les autres,")
    print(f"   l'orientation problematique est probablement systematique a cet appareil)")
    print(f"{'='*70}\n")

    # 6. PCA
    print(f"[6/7] Analyse PCA")
    X_flat = landmarks_aligned.reshape(landmarks_aligned.shape[0] * landmarks_aligned.shape[1], -1).T
    pca = PCA()
    X_pca = pca.fit_transform(X_flat)
    n_comp = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.95)[0][0] + 1
    X_pca = X_pca[:, :n_comp]
    var_exp = 100 * np.sum(pca.explained_variance_ratio_[:n_comp])
    print(f"  ✓ {n_comp} composantes ({var_exp:.1f}% variance)")

    # 7. LDA
    print(f"[7/7] Analyse Discriminante Lineaire (LDA)")
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_pca, df['groupe'])
    y_pred = lda.predict(X_pca)
    accuracy = np.mean(df['groupe'] == y_pred)
    print(f"  ✓ Taux de bonne classification: {100*accuracy:.2f}%")

    cm = confusion_matrix(df['groupe'], y_pred, labels=sorted(df['groupe'].unique()))
    cm_df = pd.DataFrame(cm, index=sorted(df['groupe'].unique()), columns=sorted(df['groupe'].unique()))

    print(f"\n{'='*70}\nReSULTATS\n{'='*70}")
    print(f"\nTaux de bonne classification: {100*accuracy:.2f}%")
    print(f"\nMatrice de confusion:\n{cm_df}")
    print(f"\nRapport par groupe:\n{classification_report(df['groupe'], y_pred, digits=3)}")

    print(f"\nGeneration des graphiques...")
    groups = sorted(df['groupe'].unique())
    n_groups = len(groups)
    fig, axes = plt.subplots(1, 2, figsize=(max(14, n_groups * 0.6), max(5, n_groups * 0.5)))

    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=True,
                xticklabels=True, yticklabels=True, annot_kws={"size": 7})
    axes[0].set_title('Matrice de confusion (LDA validation)')
    axes[0].set_ylabel('Groupe reel')
    axes[0].set_xlabel('Groupe predit')
    axes[0].tick_params(axis='x', rotation=90, labelsize=8)
    axes[0].tick_params(axis='y', rotation=0, labelsize=8)

    lda_transform = lda.transform(X_pca)
    cmap_combined = np.vstack([plt.cm.tab20(np.linspace(0, 1, 20)), plt.cm.tab20b(np.linspace(0, 1, 20))])
    colors = cmap_combined[:len(groups)]

    for i, group in enumerate(groups):
        mask = df['groupe'] == group
        y_coord = lda_transform[mask, 1] if lda_transform.shape[1] > 1 else np.zeros_like(lda_transform[mask, 0])
        axes[1].scatter(lda_transform[mask, 0], y_coord, label=group, alpha=0.6, s=50, color=colors[i])

    axes[1].set_xlabel(f"LD1 ({100*lda.explained_variance_ratio_[0]:.1f}%)")
    if lda_transform.shape[1] > 1:
        axes[1].set_ylabel(f"LD2 ({100*lda.explained_variance_ratio_[1]:.1f}%)")
    axes[1].set_title('Projection LDA')
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'lda_validation.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Graphique: {output_dir / 'lda_validation.png'}")

    cm_df.to_csv(output_dir / 'confusion_matrix.csv')
    df['Prediction'] = y_pred
    df['Correct'] = (df['groupe'] == y_pred).astype(int)
    df.to_csv(output_dir / 'predictions_detailed.csv', index=False)

    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("VALIDATION DES LANDMARKS PAR LDA\n" + "="*70 + "\n\n")
        f.write(f"Fichier TPS: {tps_file}\nFichier CSV: {csv_file}\n")
        f.write(f"Appareil filtre: {device if device else 'tous'}\n")
        f.write(f"Specimens: {landmarks.shape[2]}, Landmarks: {landmarks.shape[0]}\n")
        f.write(f"Groupes: {df['groupe'].nunique()}\n\n")
        f.write(f"PCA: {n_comp} composantes ({var_exp:.1f}% variance)\n\n")
        f.write(f"ReSULTATS LDA\n" + "-"*70 + "\n")
        f.write(f"Taux de bonne classification: {100*accuracy:.2f}%\n\n")
        f.write(cm_df.to_string())

    print(f"  ✓ Confusion matrix: {output_dir / 'confusion_matrix.csv'}")
    print(f"  ✓ Predictions: {output_dir / 'predictions_detailed.csv'}")
    print(f"  ✓ Summary: {output_dir / 'summary.txt'}")
    print(f"\n{'='*70}\n✅ Validation terminee!\n{'='*70}\n")

    return {'accuracy': accuracy, 'confusion_matrix': cm_df, 'predictions': y_pred, 'data': df}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validation des landmarks par LDA')
    parser.add_argument('--tps', required=True, help='Fichier TPS avec landmarks')
    parser.add_argument('--csv', required=True, help='Fichier CSV avec metadonnees')
    parser.add_argument('--out', default='./out/validation_results', help='Dossier de sortie')
    parser.add_argument('--id-col', default=None,
                         help="Colonne CSV correspondant a l'ID/nom TPS (auto-detecte si omis: 'name' ou 'id')")
    parser.add_argument('--device', default=None,
                         help="Filtrer sur un appareil precis (ex: P1, S3). Si omis, garde tous les appareils.")
    parser.add_argument('--device-col', default=None,
                         help="Colonne CSV contenant deja le device (sinon parse automatiquement depuis le nom)")

    args = parser.parse_args()

    results = validate_landmarks(
        args.tps, args.csv, args.out,
        id_col=args.id_col, device=args.device, device_col=args.device_col
    )