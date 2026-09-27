"""Rebuild the report chart and raw CSV from stored teacher metrics."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
base = json.loads((ROOT/'results/landmark_teacher_iterations.json').read_text(encoding='utf-8'))
rows = [('Old 320', base['historical_model_validation']['teacher_all_point_nme'])]
for round_result in base['rounds']:
    best = min(t['validation']['teacher_all_point_nme'] for t in round_result['trials'])
    rows.append((f"Ridge {round_result['train_images']}", best))
for name, label in [('landmark_hog_teacher', 'HOG 512 + local'),
                    ('landmark_hog_teacher_c2048', 'HOG 2048 + local')]:
    data = json.loads((ROOT/f'results/{name}_metrics.json').read_text(encoding='utf-8'))
    best = next(t for t in data['trials'] if t['model'] == data['selected_by_validation'])
    rows.append((label, best['validation']['teacher_all_point_nme']))
with (ROOT/'results/landmark_iterations_validation.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model_or_round', 'validation_all_point_teacher_nme'])
    writer.writerows(rows)
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(range(len(rows)), [v for _, v in rows], color=['#aaa']+['#6699bb']*3+['#448866']*2)
ax.set_xticks(range(len(rows)), [label for label, _ in rows], rotation=15)
ax.set_ylabel('All-point teacher NME (lower is better)')
ax.set_title('Fixed validation: 2,758 anime256 crops; HRNet truth')
for i, (_, value) in enumerate(rows):
    ax.text(i, value+.004, f'{value:.4f}', ha='center')
ax.set_ylim(0, .24)
fig.tight_layout()
fig.savefig(ROOT/'results/landmark_iterations_validation.png', dpi=160)
