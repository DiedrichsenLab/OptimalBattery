"""
Cross-validated task-coverage map (cortex).

Question: "Can we induce activity variations across all brain regions?"

The naive quantity -- S(v) = sum_i (X_i(v) - Xbar(v))^2 on the group library --
is biased UPWARD by noise:  E[S_obs] = S_true + sum_i sigma_i^2 > 0 always.
So it can never be zero, and cannot answer the question: any "no dead zones"
result is just the noise floor.

Fix: split SUBJECTS into two independent folds, build the library twice
(identical pipeline: recenter on rest -> average subjects -> collapse sessions
-> scale-calibrate -> merge), and take the CROSS-product instead of the square:

    S_cv(v) = sum_i ( x_i^A(v) - xbar^A(v) ) * ( x_i^B(v) - xbar^B(v) )

Noise is independent across folds, so it contributes ZERO in expectation:
E[S_cv] = S_true. S_cv can be negative -> zero is meaningful -> "we induce no
detectable variation here" becomes a statement you can actually make.

Folds are split WITHIN each dataset (MDTB subjects vs MDTB subjects, Language
vs Language), because a condition is only ever compared to its own replicate.
MDTB s1/s2 use the SAME subject split (same people in both sessions).

The scale factor is estimated ONCE on the full data and reused for both folds,
so A and B stay on a common scale and no extra noise is injected.

Averaged over N_SPLITS random splits to cut variance (stays unbiased).

Reads the already-extracted per-subject CondAll dscalars. Saves NO intermediate
libraries -- output is the cortical map only.
"""
import os
import importlib.util
from collections import OrderedDict

import numpy as np
import pandas as pd
import nibabel as nb
import matplotlib
import matplotlib.pyplot as plt

import Functional_Fusion.dataset as ds
import Functional_Fusion.atlas_map as am
import nitools as nt
from nilearn import plotting
from PIL import Image
import io

from OptimalBattery.global_config import repo_dir, data_dir

# ----------------------------- config ---------------------------------------
BASE_DIR = f'{data_dir}/FunctionalFusion_new'
SESSIONS = {'MDTB': ['s1', 's2'],
            'Language': ['localizer']}
ATLAS = 'multiatlasHCP'
CENTER_CODE = 'rest_task'
N_SPLITS = 20          # random subject splits to average over
SEED = 0
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coverage_map_cortex.png')

# reuse the library's own session-collapsing logic (module name starts with a digit)
_spec = importlib.util.spec_from_file_location(
    'tasklib', os.path.join(repo_dir, 'scripts', 'z.supp', '3.task_library.py'))
tasklib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tasklib)

import OptimalBattery.util as ut


# --------------------------- load once --------------------------------------
def load_raw():
    """Load subject-level CondAll data once per dataset/session and keep it in
    memory (row i == same participant across sessions of a dataset)."""
    raw = OrderedDict()
    nsubj = {}
    for dname, sesss in SESSIONS.items():
        for sess in sesss:
            data, info, dobj = ds.get_dataset(
                base_dir=BASE_DIR, dataset=dname, atlas=ATLAS,
                sess=f'ses-{sess}', subj=None, type='CondAll', exclude_subjects=True)
            data = np.asarray(data, dtype=np.float32)
            data[np.isnan(data)] = 0
            print(f'Loaded {dname}-{sess}: {data.shape}')
            if dname in nsubj and data.shape[0] != nsubj[dname]:
                raise ValueError(
                    f'{dname}: session {sess} has {data.shape[0]} subjects but a '
                    f'previous session had {nsubj[dname]} -- subject rows would not align.')
            nsubj[dname] = data.shape[0]
            raw[(dname, sess)] = dict(data=data, info=info,
                                      subtract_baseline=dobj.subtract_baseline)
    return raw, nsubj


# --------------------- build one library from a subject fold -----------------
def build_collapsed(raw, fold_idx):
    """Recenter on rest, average over the fold's subjects, collapse sessions.
    Mirrors make_task_library() up to the calibration step."""
    datasets = []
    for (dname, sess), R in raw.items():
        idx = fold_idx[dname]
        data = R['data'][idx]
        info = R['info'].copy()

        if R['subtract_baseline']:
            data, info = ut.recenter_data(data, info,
                                          center_full_code=CENTER_CODE, keep_center=True)
        avg_data = data.mean(axis=0)

        info = info[['task_code', 'cond_code']].copy()
        info['full_code'] = info['task_code'] + '_' + info['cond_code']
        info['source'] = f'{dname}-{sess}'

        datasets.append({'dataset': dname, 'session': sess, 'name': f'{dname}-{sess}',
                         'data': avg_data, 'info': info, 'n_subjects': len(idx)})
    return tasklib.collapse_within_dataset(datasets)


def calibrate_and_merge(collapsed, scale_factors=None):
    """Calibrate each dataset onto the root and merge -- same logic as
    make_task_library(). If scale_factors is given, reuse them instead of
    re-estimating (keeps folds on a common scale)."""
    root_data = collapsed[0]['data'].copy()
    root_info = collapsed[0]['info'].copy()
    root_weights = np.full(len(root_info), collapsed[0]['n_subjects'], dtype=float)
    used = {}

    for i in range(1, len(collapsed)):
        ds_new = collapsed[i]
        root_codes = set(root_info['full_code'])
        new_codes = set(ds_new['info']['full_code'])
        shared = [c for c in (root_codes & new_codes) if 'rest' not in c.lower()]

        if scale_factors is not None and ds_new['name'] in scale_factors:
            scale_factor = scale_factors[ds_new['name']]
        elif len(shared) == 0:
            scale_factor = 1.0
        else:
            root_mask = root_info['full_code'].isin(shared)
            new_mask = ds_new['info']['full_code'].isin(shared)
            rs = root_info[root_mask].copy(); rs['_idx'] = np.where(root_mask)[0]
            ns = ds_new['info'][new_mask].copy(); ns['_idx'] = np.where(new_mask)[0]
            merged = rs.merge(ns, on='full_code', suffixes=('_root', '_new'))
            rd = root_data[merged['_idx_root'].values]
            nd = ds_new['data'][merged['_idx_new'].values]
            num = np.sum([np.dot(rd[j], nd[j]) for j in range(len(merged))])
            den = np.sum([np.dot(nd[j], nd[j]) for j in range(len(merged))])
            scale_factor = num / den
        used[ds_new['name']] = scale_factor

        scaled_new = ds_new['data'] * scale_factor
        new_info = ds_new['info'].copy()
        new_weights = np.full(len(new_info), ds_new['n_subjects'], dtype=float)

        up_data, up_info, up_w = [], [], []
        for j, row in root_info.iterrows():
            code = row['full_code']
            match = new_info[new_info['full_code'] == code]
            if len(match) > 0:
                nj = match.index[0]
                cw = root_weights[j] + new_weights[nj]
                up_data.append((root_data[j] * root_weights[j] + scaled_new[nj] * new_weights[nj]) / cw)
                up_w.append(cw)
            else:
                up_data.append(root_data[j]); up_w.append(root_weights[j])
            up_info.append(row)
        for j, row in new_info.iterrows():
            if row['full_code'] not in root_codes:
                up_data.append(scaled_new[j]); up_info.append(row); up_w.append(new_weights[j])

        root_data = np.vstack(up_data)
        root_info = pd.DataFrame(up_info).reset_index(drop=True)
        root_weights = np.array(up_w)

    return root_data, root_info, used


def build_library(raw, fold_idx, scale_factors=None):
    data, info, used = calibrate_and_merge(build_collapsed(raw, fold_idx), scale_factors)
    return data, info['full_code'].tolist(), used


# ------------------------------ plotting ------------------------------------
def plot_cortex(S_cv, out_png=OUT_PNG, n_cond=None):
    """S_cv (P,) in multiatlasHCP grayordinates -> 4-view inflated cortex PNG.

    Displayed as the SIGNED SQUARE ROOT, sign(S_cv)*sqrt(|S_cv|/n_cond):
    S_cv is a squared-type quantity with a ~250x dynamic range, so plotting it
    raw crushes ~95% of cortex into near-white and shows only visual cortex.
    The signed sqrt compresses that ~15x and puts it back in activation units
    (the unbiased analogue of the across-task SD), while keeping the sign so
    zero -- the meaningful point -- stays readable.
    """
    template = nb.load(os.path.join(repo_dir, 'task_library', f'template_space-{ATLAS}.dscalar.nii'))
    header = nb.Cifti2Header.from_axes((nb.cifti2.ScalarAxis(['S_cv']), template.header.get_axis(1)))
    cifti = nb.Cifti2Image(dataobj=S_cv[None, :].astype(np.float64), header=header)
    surfL, surfR = [np.squeeze(np.asarray(x)) for x in nt.surf_from_cifti(cifti)]

    raw = np.concatenate([surfL, surfR])
    valid = ~np.isnan(raw)
    print(f'\n=== cross-validated task coverage (cortex, {int(valid.sum())} vertices) ===')
    print(f'  median S_cv        : {np.nanmedian(raw):.4f}')
    print(f'  vertices S_cv <= 0 : {int(np.sum(raw[valid] <= 0))} '
          f'({100*np.mean(raw[valid] <= 0):.2f}%)   <- no detectable task-induced variation')
    print(f'  dynamic range p99/p1 (positive part): '
          f'{np.nanpercentile(raw,99)/max(np.nanpercentile(raw[raw>0],1),1e-9):.0f}x')
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f'  p{p:<3}: {np.nanpercentile(raw, p):8.4f}')

    # signed sqrt for display (see docstring)
    N = n_cond if n_cond else 1
    sgn = lambda a: np.sign(a) * np.sqrt(np.abs(a) / N)
    surfL, surfR = sgn(surfL), sgn(surfR)
    both = np.concatenate([surfL, surfR])
    print(f'  -> displayed as signed sqrt: median={np.nanmedian(both):.4f}  '
          f'p99={np.nanpercentile(both,99):.4f}')

    fs32k = os.path.join(os.path.dirname(am.__file__), 'Atlases', 'tpl-fs32k')
    infl = {'left': os.path.join(fs32k, 'tpl-fs32k_hemi-L_veryinflated.surf.gii'),
            'right': os.path.join(fs32k, 'tpl-fs32k_hemi-R_veryinflated.surf.gii')}
    vmax = float(np.nanpercentile(np.abs(both), 98))

    def view(hemi, v):
        fig = plt.figure(figsize=(4.2, 4.2))
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        # deliberately no bg_on_data: blending sulc into the data paints a
        # sulcal pattern onto the map that is not in the data.
        plotting.plot_surf_stat_map(infl[hemi], stat_map=(surfL if hemi == 'left' else surfR),
                                    hemi=hemi, view=v, cmap='RdBu_r', vmax=vmax,
                                    colorbar=False, axes=ax, figure=fig)
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
        plt.close(fig); buf.seek(0)
        im = Image.open(buf).convert('RGBA')
        return im.crop(im.getbbox())

    v = {('L', 'lat'): view('left', 'lateral'), ('R', 'lat'): view('right', 'lateral'),
         ('L', 'med'): view('left', 'medial'),  ('R', 'med'): view('right', 'medial')}
    gap = 10
    cL = max(v[('L', 'lat')].width, v[('L', 'med')].width)
    cR = max(v[('R', 'lat')].width, v[('R', 'med')].width)
    r1 = max(v[('L', 'lat')].height, v[('R', 'lat')].height)
    r2 = max(v[('L', 'med')].height, v[('R', 'med')].height)

    fig = plt.figure(figsize=(1.5, (r1 + gap + r2) * 0.8 / 100))
    cax = fig.add_axes([0.06, 0.10, 0.20, 0.80])
    cb = matplotlib.colorbar.ColorbarBase(
        cax, cmap=plt.get_cmap('RdBu_r'),
        norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax), orientation='vertical')
    cb.ax.tick_params(labelsize=8); cb.set_label('cross-validated coverage  sign(S_cv)*sqrt(|S_cv|/N)', fontsize=8)
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
    plt.close(fig); buf.seek(0)
    cbi = Image.open(buf).convert('RGBA'); cbi = cbi.crop(cbi.getbbox())

    canvas = Image.new('RGBA', (cL + gap + cR + 14 + cbi.width,
                                max(r1 + gap + r2, cbi.height)), (0, 0, 0, 0))
    yoff = (canvas.height - (r1 + gap + r2)) // 2

    def place(im, cx, cy, cw, ch):
        canvas.alpha_composite(im, (cx + (cw - im.width) // 2, cy + (ch - im.height) // 2))
    place(v[('L', 'lat')], 0, yoff, cL, r1)
    place(v[('R', 'lat')], cL + gap, yoff, cR, r1)
    place(v[('L', 'med')], 0, yoff + r1 + gap, cL, r2)
    place(v[('R', 'med')], cL + gap, yoff + r1 + gap, cR, r2)
    canvas.alpha_composite(cbi, (cL + gap + cR + 14, (canvas.height - cbi.height) // 2))

    flat = Image.new('RGBA', canvas.size, (255, 255, 255, 255))
    Image.alpha_composite(flat, canvas).convert('RGB').save(out_png)
    print(f'\nsaved {out_png}')
    return out_png


# ------------------------------- main ---------------------------------------
def main():
    raw, nsubj = load_raw()
    print('\nSubjects per dataset:', nsubj)

    # scale factor estimated ONCE on the full data, reused for every fold
    all_idx = {d: np.arange(n) for d, n in nsubj.items()}
    _, codes_ref, scale_factors = build_library(raw, all_idx, scale_factors=None)
    print('Scale factors (fixed across folds):',
          {k: round(v, 4) for k, v in scale_factors.items()})
    print(f'Library: {len(codes_ref)} conditions\n')

    rng = np.random.default_rng(SEED)
    S_cv = None
    for s in range(N_SPLITS):
        foldA, foldB = {}, {}
        for d, n in nsubj.items():
            perm = rng.permutation(n)
            foldA[d], foldB[d] = perm[:n // 2], perm[n // 2:]

        libA, codesA, _ = build_library(raw, foldA, scale_factors)
        libB, codesB, _ = build_library(raw, foldB, scale_factors)

        # align B to A's condition order
        if codesA != codesB:
            order = [codesB.index(c) for c in codesA]
            libB = libB[order]

        xA = libA - libA.mean(axis=0, keepdims=True)   # centre across tasks, per fold
        xB = libB - libB.mean(axis=0, keepdims=True)
        s_cv = np.sum(xA * xB, axis=0)                 # (P,) unbiased signal variance

        S_cv = s_cv if S_cv is None else S_cv + s_cv
        print(f'  split {s+1:>2}/{N_SPLITS}  '
              f'A={[len(foldA[d]) for d in nsubj]} B={[len(foldB[d]) for d in nsubj]}  '
              f'median S_cv={np.median(s_cv):.4f}  frac<=0={np.mean(s_cv <= 0):.3f}')
    S_cv /= N_SPLITS

    out = plot_cortex(S_cv, n_cond=len(codes_ref))
    try:
        plt.figure(figsize=(11, 7)); plt.imshow(Image.open(out)); plt.axis('off'); plt.show()
    except Exception:
        pass


if __name__ == '__main__':
    main()
