import numpy as np
np.random.seed(42)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil, os

# ── Parameters ───────────────────────────────────────────────────────────────
N_BB1 = 100; N_BB2 = 100; N_BB3 = 100  # building blocks per cycle
N_COMPOUNDS = N_BB1 * N_BB2 * N_BB3    # 1,000,000 compounds
N_ROUNDS = 3

# ── Simulate DEL synthesis ───────────────────────────────────────────────────
# Each compound identified by (bb1, bb2, bb3) indices
# Sample 100k compounds for efficiency
N_SAMPLE = 100000
bb1_idx = np.random.randint(0, N_BB1, N_SAMPLE)
bb2_idx = np.random.randint(0, N_BB2, N_SAMPLE)
bb3_idx = np.random.randint(0, N_BB3, N_SAMPLE)

# ── Affinity selection simulation ────────────────────────────────────────────
# True binders: ~0.1% of library (pharmacophore model)
# Pharmacophore: specific bb1 (5 scaffolds) + bb2 (10 groups) + bb3 (8 groups)
active_bb1 = np.array([3, 17, 42, 68, 91])
active_bb2 = np.array([5, 12, 23, 34, 45, 56, 67, 78, 89, 95])
active_bb3 = np.array([7, 19, 31, 43, 55, 67, 79, 91])

is_active = (np.isin(bb1_idx, active_bb1) &
             np.isin(bb2_idx, active_bb2) &
             np.isin(bb3_idx, active_bb3))

# Enrichment per round: active compounds enriched ~10x per round
base_count = np.random.poisson(5, N_SAMPLE)  # baseline reads
target_count = base_count.copy()
for r in range(N_ROUNDS):
    target_count[is_active] = np.random.poisson(50 * (10**r), is_active.sum())
    target_count[~is_active] = np.random.poisson(5, (~is_active).sum())

# No-target control
control_count = np.random.poisson(5, N_SAMPLE)

# ── Enrichment scoring ───────────────────────────────────────────────────────
enrichment = np.log2((target_count + 1) / (control_count + 1))

# ── Hit identification (enrichment > 3σ) ─────────────────────────────────────
enrich_mean = enrichment.mean()
enrich_std  = enrichment.std()
hit_threshold = enrich_mean + 3 * enrich_std
hits = enrichment > hit_threshold
n_hits = hits.sum()
hit_rate = n_hits / N_SAMPLE

# ── Chemical space analysis ──────────────────────────────────────────────────
# Simulate MW and LogP from building block properties
mw_bb1 = np.random.normal(150, 30, N_BB1)
mw_bb2 = np.random.normal(100, 20, N_BB2)
mw_bb3 = np.random.normal(80, 15, N_BB3)
logp_bb1 = np.random.normal(1.5, 0.8, N_BB1)
logp_bb2 = np.random.normal(0.8, 0.6, N_BB2)
logp_bb3 = np.random.normal(0.5, 0.5, N_BB3)

mw_compounds = mw_bb1[bb1_idx] + mw_bb2[bb2_idx] + mw_bb3[bb3_idx] + 50  # linker
logp_compounds = logp_bb1[bb1_idx] + logp_bb2[bb2_idx] + logp_bb3[bb3_idx]

# ── Scaffold diversity ───────────────────────────────────────────────────────
scaffold_counts = np.bincount(bb1_idx, minlength=N_BB1)
scaffold_diversity = -np.sum((scaffold_counts/scaffold_counts.sum()) *
                              np.log(scaffold_counts/scaffold_counts.sum() + 1e-9))

# ── SAR from enrichment patterns ─────────────────────────────────────────────
# Mean enrichment per bb1 scaffold
sar_bb1 = np.array([enrichment[bb1_idx == i].mean() if (bb1_idx == i).sum() > 0 else 0
                     for i in range(N_BB1)])
sar_bb2 = np.array([enrichment[bb2_idx == i].mean() if (bb2_idx == i).sum() > 0 else 0
                     for i in range(N_BB2)])

# ── Selection round comparison ───────────────────────────────────────────────
round_enrichments = []
for r in range(N_ROUNDS):
    tc = np.random.poisson(5 * (3**r), N_SAMPLE)
    tc[is_active] = np.random.poisson(50 * (10**r), is_active.sum())
    enr = np.log2((tc + 1) / (control_count + 1))
    round_enrichments.append(enr)

# ── False positive filtering ─────────────────────────────────────────────────
# FP: high enrichment but not truly active (matrix effect)
fp_mask = hits & ~is_active
fp_rate = fp_mask.sum() / (hits.sum() + 1e-9)
true_positive_rate = (hits & is_active).sum() / (is_active.sum() + 1e-9)

# ── Hit cluster analysis ─────────────────────────────────────────────────────
hit_mw   = mw_compounds[hits]
hit_logp = logp_compounds[hits]

# ── Key results ──────────────────────────────────────────────────────────────
print("=== DNAEncodedLibraryEngine Key Results ===")
print(f"Library size: {N_COMPOUNDS:,} compounds ({N_SAMPLE:,} sampled)")
print(f"True actives in sample: {is_active.sum()} ({is_active.mean()*100:.2f}%)")
print(f"Hit threshold: {hit_threshold:.2f} (mean+3σ)")
print(f"Hits identified: {n_hits} ({hit_rate*100:.2f}%)")
print(f"True positive rate: {true_positive_rate*100:.1f}%")
print(f"False positive rate: {fp_rate*100:.1f}%")
print(f"Scaffold diversity (entropy): {scaffold_diversity:.2f}")

# ── Dashboard ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.patch.set_facecolor('#0d1117')
fig.suptitle('DNA-Encoded Library Engine — DEL Selection & Hit Analysis',
             color='white', fontsize=16, fontweight='bold', y=0.98)

COLORS = ['#58a6ff','#3fb950','#f78166','#d2a8ff','#ffa657','#79c0ff','#56d364','#ff7b72']

# Panel 1: Enrichment distribution
ax = axes[0, 0]; ax.set_facecolor('#161b22')
ax.hist(enrichment[~hits], bins=60, color='#58a6ff', alpha=0.7, label='Non-hits', density=True)
ax.hist(enrichment[hits],  bins=30, color='#f78166', alpha=0.9, label=f'Hits (n={n_hits})', density=True)
ax.axvline(hit_threshold, color='#ffa657', ls='--', lw=2, label=f'Threshold={hit_threshold:.1f}')
ax.set_xlabel('log2 Enrichment', color='white'); ax.set_ylabel('Density', color='white')
ax.set_title('Enrichment Score Distribution', color='white', fontweight='bold')
ax.tick_params(colors='white'); ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 2: Hit identification (volcano-style)
ax = axes[0, 1]; ax.set_facecolor('#161b22')
ax.scatter(enrichment[~hits], np.random.uniform(0, 1, (~hits).sum()),
           color='#58a6ff', s=1, alpha=0.3)
ax.scatter(enrichment[hits], np.random.uniform(0, 1, hits.sum()),
           color='#f78166', s=5, alpha=0.8, label=f'Hits (n={n_hits})')
ax.axvline(hit_threshold, color='#ffa657', ls='--', lw=1.5)
ax.set_xlabel('log2 Enrichment', color='white'); ax.set_ylabel('Random Jitter', color='white')
ax.set_title('Hit Identification', color='white', fontweight='bold')
ax.tick_params(colors='white'); ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 3: Chemical space (MW vs LogP)
ax = axes[0, 2]; ax.set_facecolor('#161b22')
ax.scatter(mw_compounds[~hits], logp_compounds[~hits], color='#58a6ff', s=1, alpha=0.2)
ax.scatter(mw_compounds[hits],  logp_compounds[hits],  color='#f78166', s=10, alpha=0.8, label='Hits')
ax.axhline(5, color='#ffa657', ls='--', lw=1, label='LogP=5 (Ro5)')
ax.axvline(500, color='#3fb950', ls='--', lw=1, label='MW=500 (Ro5)')
ax.set_xlabel('Molecular Weight (Da)', color='white'); ax.set_ylabel('LogP', color='white')
ax.set_title('Chemical Space (MW vs LogP)', color='white', fontweight='bold')
ax.tick_params(colors='white'); ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 4: Scaffold diversity
ax = axes[1, 0]; ax.set_facecolor('#161b22')
sorted_scaffolds = np.sort(scaffold_counts)[::-1]
ax.bar(range(N_BB1), sorted_scaffolds, color='#d2a8ff', alpha=0.8, edgecolor='none')
ax.set_xlabel('Scaffold Rank', color='white'); ax.set_ylabel('Compound Count', color='white')
ax.set_title(f'Scaffold Diversity (H={scaffold_diversity:.2f})', color='white', fontweight='bold')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 5: SAR heatmap (bb1 × bb2 mean enrichment, 20×20 subset)
ax = axes[1, 1]; ax.set_facecolor('#161b22')
sar_matrix = np.zeros((20, 20))
for i in range(20):
    for j in range(20):
        mask = (bb1_idx == i) & (bb2_idx == j)
        if mask.sum() > 0:
            sar_matrix[i, j] = enrichment[mask].mean()
im = ax.imshow(sar_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
cb = plt.colorbar(im, ax=ax, label='Mean Enrichment')
cb.ax.yaxis.label.set_color('white'); cb.ax.tick_params(colors='white')
ax.set_xlabel('BB2 Index', color='white'); ax.set_ylabel('BB1 Index', color='white')
ax.set_title('SAR Heatmap (BB1 x BB2)', color='white', fontweight='bold')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 6: Selection round comparison
ax = axes[1, 2]; ax.set_facecolor('#161b22')
for r, (enr, col) in enumerate(zip(round_enrichments, ['#58a6ff','#3fb950','#f78166'])):
    ax.hist(enr, bins=50, alpha=0.6, color=col, label=f'Round {r+1}', density=True)
ax.set_xlabel('log2 Enrichment', color='white'); ax.set_ylabel('Density', color='white')
ax.set_title('Selection Round Comparison', color='white', fontweight='bold')
ax.tick_params(colors='white'); ax.legend(facecolor='#21262d', labelcolor='white', fontsize=8)
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 7: False positive rate
ax = axes[2, 0]; ax.set_facecolor('#161b22')
categories = ['True Positives','False Positives','True Negatives','False Negatives']
tp = (hits & is_active).sum()
fp_n = (hits & ~is_active).sum()
tn = (~hits & ~is_active).sum()
fn = (~hits & is_active).sum()
values = [tp, fp_n, tn, fn]
colors_fp = ['#3fb950','#f78166','#58a6ff','#ffa657']
ax.bar(range(4), values, color=colors_fp, alpha=0.85, edgecolor='#30363d')
ax.set_xticks(range(4)); ax.set_xticklabels(categories, color='white', rotation=15, fontsize=8)
ax.set_ylabel('Count', color='white')
ax.set_title(f'Classification (FPR={fp_rate*100:.1f}%)', color='white', fontweight='bold')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 8: Hit cluster analysis
ax = axes[2, 1]; ax.set_facecolor('#161b22')
if len(hit_mw) > 0:
    sc = ax.scatter(hit_mw, hit_logp, c=enrichment[hits], cmap='hot', s=30, alpha=0.8)
    cb = plt.colorbar(sc, ax=ax, label='Enrichment')
    cb.ax.yaxis.label.set_color('white'); cb.ax.tick_params(colors='white')
ax.set_xlabel('MW (Da)', color='white'); ax.set_ylabel('LogP', color='white')
ax.set_title('Hit Cluster Analysis', color='white', fontweight='bold')
ax.tick_params(colors='white')
for sp in ax.spines.values(): sp.set_color('#30363d')

# Panel 9: Summary
ax = axes[2, 2]; ax.set_facecolor('#161b22'); ax.axis('off')
ax.set_title('Summary Statistics', color='white', fontweight='bold')
summary_lines = [
    ('Library Size', f'{N_COMPOUNDS:,}'),
    ('Sampled', f'{N_SAMPLE:,}'),
    ('Selection Rounds', f'{N_ROUNDS}'),
    ('Hit Threshold', f'{hit_threshold:.2f} (mean+3σ)'),
    ('Hits Identified', f'{n_hits} ({hit_rate*100:.2f}%)'),
    ('True Positive Rate', f'{true_positive_rate*100:.1f}%'),
    ('False Positive Rate', f'{fp_rate*100:.1f}%'),
    ('Scaffold Diversity', f'{scaffold_diversity:.2f} bits'),
    ('Ro5 Compliant Hits', f'{np.sum((hit_mw<500)&(hit_logp<5))}'),
]
for idx, (k, v) in enumerate(summary_lines):
    ax.text(0.05, 0.88 - idx*0.10, k, color='#8b949e', fontsize=10, transform=ax.transAxes)
    ax.text(0.65, 0.88 - idx*0.10, v, color='#58a6ff', fontsize=10, fontweight='bold', transform=ax.transAxes)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/mnt/shared-workspace/shared/dna_encoded_library_engine_dashboard.png',
            dpi=100, bbox_inches='tight', facecolor='#0d1117')
plt.close()
shutil.copy(__file__, '/mnt/shared-workspace/shared/dna_encoded_library_engine.py')
print("Dashboard saved.")
