"""Render honest, incremental run-status and evaluator-audit figures.

CLI: python progress.py --run-dir ETC/runs/main --audit-dir ETC/data/paper/audit
Only PNG/PDF figures are written to Review. Missing run artifacts stay pending.
Accepted status shape: {stages: {P1: {status, completed, total, elapsed_seconds,
eta_seconds}}, counters: {input_sequences, inference_seconds}}. Stages may also
be a list, or called phases. Plan stages may specify planned_edits and
estimated_seconds. Measured benchmark timing is a fallback for extrapolation,
never evidence that a later experiment ran.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'artifacts' / 'Review'
DEFAULT_STAGES = (
    ('P0', 'Evaluator audit'),
    ('P1', 'Seven-algorithm tournament'),
    ('E1', 'Frozen held-out performance'),
    ('E2', '16-hop semantic lifetime'),
    ('E3', 'Theory diagnostics'),
    ('E4', 'Ablation and cost'),
)
BLUE = '#2375AA'
GREEN = '#27836A'
GRAY = '#DCE3E9'
ORANGE = '#D98B24'
RED = '#B54142'


def read_json(path):
    """A writer may be replacing an artifact; a missing snapshot is not fatal."""
    try:
        with Path(path).open() as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def number(value, default=None):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def first_number(data, names, default=None):
    for name in names:
        value = number(data.get(name))
        if value is not None:
            return value
    return default


def duration(seconds):
    seconds = number(seconds)
    if seconds is None:
        return 'not measured'
    if seconds < 60:
        return f'{max(0, seconds):.0f} s'
    if seconds < 3600:
        return f'{seconds / 60:.1f} min'
    return f'{seconds / 3600:.1f} h'


def stage_map(data):
    raw = data.get('stages', data.get('phases', data.get('phase_estimates', {})))
    if isinstance(raw, list):
        result = {}
        for i, item in enumerate(raw):
            if isinstance(item, dict):
                key = str(item.get('id', item.get('stage', item.get('phase', item.get('name', i)))))
                result[key] = item
        return result
    if isinstance(raw, dict):
        return {str(k): (v if isinstance(v, dict) else {'status': v}) for k, v in raw.items()}
    return {}


def save_figure(fig, out_dir, stem):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ('.png', '.pdf'):
        target = out_dir / (stem + suffix)
        temp = out_dir / ('.' + stem + f'.{os.getpid()}.tmp' + suffix)
        fig.savefig(temp, dpi=190, bbox_inches='tight', facecolor='white')
        temp.replace(target)
        paths.append(str(target))
    plt.close(fig)
    return paths


def configure():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 12, 'axes.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white',
    })


def render_progress(status_path, plan_path, out_dir=REVIEW, benchmark_path=None):
    configure()
    status_path, plan_path = Path(status_path), Path(plan_path)
    status, plan = read_json(status_path), read_json(plan_path)
    benchmark = read_json(benchmark_path or status_path.parent / 'benchmark.json')
    measured = status.get('counters', status.get('metrics', {}))
    if not isinstance(measured, dict):
        measured = {}
    sequences = first_number(measured, ['input_sequences', 'generated_sequences', 'unique_edits'])
    infer_seconds = first_number(measured, ['inference_seconds', 'generation_seconds'])
    rate = sequences / infer_seconds if sequences and infer_seconds and infer_seconds > 0 else None
    rate_label = 'run inference'
    if rate is None:
        counters = benchmark.get('counters', {})
        if not isinstance(counters, dict):
            counters = {}
        benchmark_n = first_number(counters, ['input_sequences'], number(benchmark.get('states')))
        benchmark_s = first_number(counters, ['inference_seconds'], number(benchmark.get('wall_seconds')))
        if benchmark_n and benchmark_s and benchmark_s > 0:
            rate = benchmark_n / benchmark_s
            rate_label = f'initial benchmark ({benchmark_n:.0f} edits)'
    runtime, planned = stage_map(status), stage_map(plan)
    # Parent runner persists completed phase timings and one current-phase snapshot.
    timings = status.get('timings', read_json(status_path.parent/'timings.json'))
    if isinstance(timings, dict):
        for phase, observed in timings.items():
            if isinstance(observed, dict):
                runtime[phase] = {**observed, 'status':'complete',
                                  'completed': observed.get('state_count')}
    current = str(status.get('stage', status.get('phase', status.get('current_stage', ''))))
    if current and current not in runtime:
        runtime[current] = {'status': 'failed' if status.get('status_detail')=='stage_failed' else 'running',
                            'completed': status.get('phase_done'), 'total':status.get('phase_total'),
                            'elapsed_seconds':status.get('phase_elapsed_seconds'),
                            'eta_seconds':status.get('phase_remaining_seconds')}
    # Two plan entries summarize several concrete phases. Completion requires their
    # final persisted artifact, not simply the first subphase finishing.
    aggregations = {
        'E4_ablation_and_cost': ('E4_', 'E4_cost_and_search_scaling'),
        'E1_external_transfer': ('E1_external_', None),
    }
    for key, (prefix, final_phase) in aggregations.items():
        if key not in planned:
            continue
        pieces = [(k,v) for k,v in runtime.items() if k.startswith(prefix) and k != key]
        done_pieces = [v for k,v in pieces if v.get('status')=='complete']
        active_piece = next((v for k,v in pieces if k == current and v.get('status')!='complete'), None)
        complete = ((final_phase in timings) if final_phase and isinstance(timings,dict) else
                    (status_path.parent/'external/summary.json').exists())
        if pieces or complete:
            runtime[key] = {'status':'complete' if complete else 'running',
                'completed':sum(number(v.get('completed'),0) for _,v in pieces),
                'elapsed_seconds':sum(first_number(v,['elapsed_seconds','wall_seconds'],0) for _,v in pieces)}
            if active_piece and active_piece.get('status')=='failed':
                runtime[key]['status']='failed'
    keys = list(planned) if planned else list(runtime)
    if status.get('status')=='complete':
        for key in keys:
            runtime[key]={**runtime.get(key,{}),'status':'complete'}
    if not keys:
        keys = [key for key, _ in DEFAULT_STAGES]
    names = dict(DEFAULT_STAGES)
    names.update({'P0_evaluator_audit':'Evaluator audit','P1_algorithm_tournament':'Seven-algorithm tournament',
                  'E1_final_performance':'Frozen held-out performance','E2_lifetime':'16-hop semantic lifetime',
                  'E3_theory_diagnostics':'Theory diagnostics','E4_ablation_and_cost':'Ablation and cost',
                  'E1_external_transfer':'External retention'})
    current = str(status.get('stage', status.get('phase', status.get('current_stage', ''))))
    rows = []
    for key in keys:
        observed, expected = runtime.get(key, {}), planned.get(key, {})
        label = str(observed.get('label', observed.get('name', expected.get('label', expected.get('name', names.get(key, key))))))
        state = str(observed.get('status', 'pending')).lower()
        if not observed and current.lower() == key.lower():
            state = str(status.get('status', 'running')).lower()
        done = first_number(observed, ['completed', 'completed_edits', 'states_done', 'done', 'completed_units'])
        total = first_number(observed, ['total', 'planned_edits', 'total_edits', 'total_units'])
        if total is None:
            total = first_number(expected, ['total', 'planned_edits', 'planned_states', 'total_edits', 'edits', 'unique_edits', 'total_units'])
        elapsed = first_number(observed, ['elapsed_seconds', 'wall_seconds', 'seconds'])
        eta = first_number(observed, ['eta_seconds', 'remaining_seconds'])
        estimate = first_number(expected, ['estimated_seconds', 'estimate_seconds', 'seconds'])
        if estimate is None:
            minutes = first_number(expected, ['estimated_minutes', 'minutes'])
            estimate = minutes * 60 if minutes is not None else None
        range_low = number(expected.get('minutes_low'))
        range_high = number(expected.get('minutes_high'))
        complete = state in ('complete', 'completed', 'done', 'passed')
        failed = state in ('failed', 'error', 'blocked')
        active = state in ('running', 'active', 'in_progress', 'in-progress')
        fraction = 1.0 if complete else min(1., max(0., done / total)) if done is not None and total else 0.
        if eta is None and range_low is not None and range_high is not None and not complete and not failed:
            eta = (range_low + range_high) * 30 * (1-fraction)
        if eta is None and total is not None and rate and not complete and not failed:
            eta = max(0., total - (done or 0.)) / rate
        if eta is None and estimate is not None and not complete and not failed:
            eta = max(0., estimate * (1 - fraction))
        # A completed flag is independent evidence; missing totals stay unspecified.
        progress = f'{done:,.0f} / {total:,.0f}' if done is not None and total is not None else (
            f'{total:,.0f} planned' if total is not None else 'no count snapshot')
        if complete and done is not None and total is not None and done != total:
            progress = f'{done:,.0f} completed'
        if total == 0 and not complete:
            progress='CPU analysis planned'
        detail = ('measured ' + duration(elapsed)) if complete and elapsed is not None else (
            'remaining ~' + duration(eta) if eta is not None else ('measured ' + duration(elapsed) if elapsed is not None else 'ETA not measured'))
        if not active and not complete and not failed and range_low is not None and range_high is not None:
            detail=f'plan ~{range_low:g}–{range_high:g} min'
        rows.append(dict(key=key.split('_')[0], label=label, state=state, fraction=fraction,
                         color=GREEN if complete else RED if failed else BLUE if active else GRAY,
                         progress=progress, detail=detail, complete=complete, eta=eta, failed=failed,
                         eta_low=(range_low*60*(1-fraction) if not active and range_low is not None else eta),
                         eta_high=(range_high*60*(1-fraction) if not active and range_high is not None else eta)))
    height = max(4.8, 1.9 + .61 * len(rows))
    fig, ax = plt.subplots(figsize=(11.5, height))
    fig.subplots_adjust(left=.32, right=.75, top=.79, bottom=.19)
    y = np.arange(len(rows))
    ax.barh(y, [1] * len(rows), height=.46, color='#EDF1F4')
    for i, row in enumerate(rows):
        if row['fraction']:
            ax.barh(i, row['fraction'], height=.46, color=row['color'])
        elif row['state'] in ('running', 'active', 'in_progress', 'in-progress'):
            # Small marker identifies a running stage; it is not a progress estimate.
            ax.scatter([0], [i], color=BLUE, s=38, zorder=3)
        text = row['state'].replace('_', ' ')
        ax.text(1.035, i - .09, text, va='center', color=row['color'] if row['color'] != GRAY else '#657280', weight='bold', fontsize=10)
        ax.text(1.035, i + .18, row['detail'], va='center', fontsize=9, color='#526170')
        if row['progress'] != 'no count snapshot':
            ax.text(.02, i, row['progress'], va='center', fontsize=9,
                    color='white' if row['fraction'] > .36 else '#233541')
    ax.set_yticks(y, [f"{r['key']}  {r['label']}" for r in rows])
    ax.tick_params(axis='y', length=0, pad=12)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .5, 1], ['0%', '50%', '100%'])
    ax.tick_params(axis='x', length=0, colors='#6D7984')
    for side in ('left', 'bottom'):
        ax.spines[side].set_visible(False)
    fig.suptitle('Telephone Game | experiment progress', x=.04, ha='left', y=.97, fontsize=18, weight='bold', color='#1D3444')
    if rate is not None:
        throughput = f'{rate:.2f} unique edits/s · {rate_label}'
    else:
        throughput = 'Throughput has not been measured'
    fig.text(.04, .875, throughput, fontsize=10, color='#435F72')
    if current:
        current_done=number(status.get('phase_done'))
        current_total=number(status.get('phase_total'))
        current_label=f'Current phase: {current.replace("_"," ")}'
        if current_done is not None and current_total:
            current_label+=f' · {current_done:,.0f}/{current_total:,.0f} states'
        if status.get('status')=='complete':
            current_label='All registered experiments completed'
        fig.text(.04, .825, current_label, fontsize=9, color='#657280')
    known_remaining = [r['eta'] for r in rows if not r['complete'] and not r['failed'] and r['eta'] is not None]
    unknown = any(not r['complete'] and r['eta'] is None for r in rows)
    if known_remaining:
        low=sum(r['eta_low'] or 0 for r in rows if not r['complete'] and not r['failed'])
        high=sum(r['eta_high'] or 0 for r in rows if not r['complete'] and not r['failed'])
        if current not in planned:
            extra=number(status.get('phase_remaining_seconds'),0)
            low+=extra; high+=extra
        prefix = 'Known stages remaining ~' if unknown else 'Estimated remaining ~'
        interval=(f'{low/60:.0f}–{high/60:.0f} min' if abs(high-low)>60 else duration(high))
        fig.text(.68, .875, prefix + interval, fontsize=10, color='#435F72')
    updated = datetime.now(ZoneInfo('America/New_York')).strftime('%b %d, %Y %I:%M %p %Z')
    note = 'Current-stage ETA uses measured progress; pending-stage ranges are the initial timing plan.'
    if not status:
        note = 'No run-status snapshot yet. The initial benchmark is not a completed experiment.'
    fig.text(.04, .075, note, fontsize=9, color='#526170')
    fig.text(.04, .035, f'Updated {updated} · stage completion comes from persisted run artifacts', fontsize=8.5, color='#6D7984')
    return save_figure(fig, out_dir, '00_run_progress')


def wilson(correct, total, z=1.96):
    if not total:
        return (0., 1.)
    p = correct / total
    den = 1 + z * z / total
    middle = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return max(0., middle - half), min(1., middle + half)


def render_audit(audit_dir, out_dir=REVIEW):
    configure()
    audit_dir = Path(audit_dir)
    report = read_json(audit_dir / 'evaluator_report.json')
    if not report:
        return []
    by_type = report.get('by_type', {})
    if not by_type:
        return []
    items = sorted(by_type.items(), key=lambda item: (item[1].get('accuracy', 0), item[0]))
    fig, (ax, matrix_ax) = plt.subplots(1, 2, figsize=(13.2, max(5.8, 2.3 + .31 * len(items))), gridspec_kw={'width_ratios': [1.35, 1]})
    fig.subplots_adjust(left=.19, right=.96, top=.80, bottom=.20, wspace=.46)
    ypos = np.arange(len(items))
    for i, (label, entry) in enumerate(items):
        n, accuracy = int(entry.get('n', 0)), number(entry.get('accuracy'), 0)
        lo, hi = wilson(round(n * accuracy), n)
        ax.plot([lo, hi], [i, i], color='#A5B8C5', lw=3, solid_capstyle='round')
        ax.scatter([accuracy], [i], color=GREEN if accuracy >= .8 else RED, s=42, zorder=3)
        ax.text(1.025, i, f'{accuracy:.0%}  (n={n})', va='center', fontsize=9)
    ax.set_yticks(ypos, [label.replace('_', ' ') for label, _ in items], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, .25, .5, .75, 1], ['0%', '25%', '50%', '75%', '100%'])
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Case-label accuracy (Wilson 95% interval)', labelpad=12)
    ax.set_title('Controlled counterfactual cases', loc='left', pad=12)
    labels = ['preserved', 'missing', 'altered', 'ambiguous']
    conf = report.get('confusion', {})
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for key, count in conf.items():
        if '->' in key:
            actual, predicted = key.split('->', 1)
            if actual in labels and predicted in labels:
                matrix[labels.index(actual), labels.index(predicted)] += int(count)
    if not matrix.sum():
        try:
            with (audit_dir / 'evaluator_audit.csv').open(newline='') as handle:
                for row in csv.DictReader(handle):
                    if row['expected'] in labels and row['predicted'] in labels:
                        matrix[labels.index(row['expected']), labels.index(row['predicted'])] += 1
        except OSError:
            pass
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    matrix_ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    for i in range(4):
        for j in range(4):
            matrix_ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if normalized[i, j] > .55 else '#394D5B', fontsize=12)
    matrix_ax.set_xticks(range(4), labels, rotation=35, ha='right', fontsize=9)
    matrix_ax.set_yticks(range(4), labels, fontsize=9)
    matrix_ax.set_xlabel('Predicted label')
    matrix_ax.set_ylabel('Expected label')
    matrix_ax.set_title('Audit confusion matrix (case counts)', pad=12, fontsize=11)
    for spine in matrix_ax.spines.values():
        spine.set_visible(False)
    fig.suptitle('Telephone Game | evaluator validation', x=.04, y=.97, ha='left', fontsize=18, weight='bold', color='#1D3444')
    n = report.get('cases', int(matrix.sum()))
    p, r = number(report.get('preserved_precision')), number(report.get('preserved_recall'))
    metrics = f'{n} generated audit cases'
    if p is not None and r is not None:
        metrics += f' · preserved precision {p:.1%} · recall {r:.1%}'
    fig.text(.04, .865, metrics, fontsize=11, color='#435F72')
    fig.text(.04, .075, 'Scope: known-template perturbations. These checks do not validate unrestricted model paraphrases or external data.', fontsize=9, color='#526170')
    fig.text(.04, .035, 'The intervals summarize audit-case counts; related cases share generated source templates.', fontsize=8.5, color='#6D7984')
    return save_figure(fig, out_dir, '00_evaluator_audit')


def update_review(run_dir=ROOT/'runs/main', review_dir=REVIEW, audit_dir=ROOT/'data/paper/audit'):
    run_dir = Path(run_dir)
    return render_progress(run_dir/'status.json', run_dir/'timing_plan.json', review_dir) + render_audit(audit_dir, review_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', '--run', type=Path, default=ROOT/'runs/main')
    parser.add_argument('--audit-dir', type=Path, default=ROOT/'data/paper/audit')
    parser.add_argument('--review-dir', '--review', type=Path, default=REVIEW)
    parser.add_argument('--status', type=Path, help='Optional explicit status JSON path')
    parser.add_argument('--timing-plan', type=Path, help='Optional explicit timing-plan JSON path')
    parser.add_argument('--watch-seconds', type=float, default=0, help='Positive value refreshes repeatedly; Ctrl-C stops')
    args = parser.parse_args()
    while True:
        paths = render_progress(args.status or args.run_dir/'status.json', args.timing_plan or args.run_dir/'timing_plan.json', args.review_dir)
        paths += render_audit(args.audit_dir, args.review_dir)
        print(json.dumps({'figures': paths}), flush=True)
        if args.watch_seconds <= 0:
            break
        time.sleep(max(1, args.watch_seconds))


if __name__ == '__main__':
    main()
