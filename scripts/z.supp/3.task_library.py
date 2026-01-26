import numpy as np
import pandas as pd
import Functional_Fusion.dataset as ds
import OptimalBattery.util as ut
import nibabel as nb
import Functional_Fusion.atlas_map as am
from OptimalBattery.global_config import repo_dir, data_dir
import os
from collections import OrderedDict

def make_task_library(base_dir, sessions_to_use, atlas='MNISymC3'):
    """
    Load datasets, recenter around rest, calibrate scaling using shared conditions,
    and merge into a single library.

    Args:
        base_dir: base directory for datasets (functional_fusion)
        sessions_to_use: dict of dataset names to list of sessions
        * Order matters: every dataset/session here needs to share at least one task + rest with the previous dataset
        atlas: atlas name

    Returns:
        final_data, final_info
    """

    # Load all datasets, recenter around rest, store info and n_subs
    datasets = []
    for dataset_name in sessions_to_use.keys():
        for sess in sessions_to_use[dataset_name]:
            data, info, dataset_obj = ds.get_dataset(
                base_dir=base_dir, dataset=dataset_name, atlas=atlas,
                sess=f'ses-{sess}', subj=None, type='CondAll', exclude_subjects=True
            )
            data[np.isnan(data)] = 0
            n_subjects = data.shape[0]

            print(f'Loaded: {dataset_name}-{sess}, shape: {data.shape}')

            # Recenter around rest if baseline already subtracted
            if dataset_obj.subtract_baseline:
                center_code = 'rest_task'
                print(f'  Recentering around {center_code}')
                data, info = ut.recenter_data(data, info, center_full_code=center_code, keep_center=True)

            # Average across subjects
            avg_data = data.mean(axis=0)

            # Keep essential columns
            info = info[['task_code', 'cond_code']].copy()
            info['full_code'] = info['task_code'] + '_' + info['cond_code']
            info['source'] = f'{dataset_name}-{sess}'

            datasets.append({
                'dataset': dataset_name,  
                'session': sess,
                'name': f'{dataset_name}-{sess}',
                'data': avg_data,
                'info': info,
                'n_subjects': n_subjects
            })

    # Collapse sessions within each dataset 
    collapsed_datasets = collapse_within_dataset(datasets)

    # Calibrate and merge datasets one by one, first dataset is the root
    root_data = collapsed_datasets[0]['data'].copy()
    root_info = collapsed_datasets[0]['info'].copy()
    root_weights = np.full(len(root_info), collapsed_datasets[0]['n_subjects'], dtype=float)
    print(f"Root dataset: {collapsed_datasets[0]['name']} ({collapsed_datasets[0]['n_subjects']} subjects)")

    # Calibrate each subsequent dataset to root
    for i in range(1, len(collapsed_datasets)):
        ds_new = collapsed_datasets[i]
        ds_new['info']['full_code'] = ds_new['info']['full_code']
        print(f" Calibrating {ds_new['name']} ({ds_new['n_subjects']} subjects)")

        # Find shared conditions (exclude rest - it's zeroed out after recentering)
        root_codes = set(root_info['full_code'])
        new_codes = set(ds_new['info']['full_code'])
        shared_codes = [c for c in (root_codes & new_codes) if 'rest' not in c.lower()]

        if len(shared_codes) == 0:
            print(f"  warning: no shared conditions with root, skipping calibration")
            scale_factor = 1.0
        else:
            print(f"  Shared conditions: {len(shared_codes)}")

            # Get shared condition data
            root_mask = root_info['full_code'].isin(shared_codes)
            new_mask = ds_new['info']['full_code'].isin(shared_codes)

            # Align order by full_code
            root_shared = root_info[root_mask].copy()
            new_shared = ds_new['info'][new_mask].copy()
            root_shared['_idx'] = np.where(root_mask)[0]
            new_shared['_idx'] = np.where(new_mask)[0]

            merged = root_shared.merge(new_shared, on='full_code', suffixes=('_root', '_new'))
            root_idx = merged['_idx_root'].values
            new_idx = merged['_idx_new'].values

            root_shared_data = root_data[root_idx]
            new_shared_data = ds_new['data'][new_idx]

            # Check correlations (sanity check)
            print("  Correlations per shared condition:")
            correlations = []
            for j, code in enumerate(merged['full_code'].values):
                r = np.corrcoef(root_shared_data[j], new_shared_data[j])[0, 1]
                correlations.append(r)
                print(f"    {code}: r={r:.3f}")
            mean_corr = np.mean(correlations)
            print(f"  Mean correlation: {mean_corr:.3f}")

            if mean_corr < 0.3:
                print("  warning: low correlations, calibration may be unreliable")

            # Compute individual scaling factors per condition (sanity check)
            print("  Individual scaling factors:")
            individual_scales = []
            for j, code in enumerate(merged['full_code'].values):
                s = np.dot(root_shared_data[j], new_shared_data[j]) / np.dot(new_shared_data[j], new_shared_data[j])
                individual_scales.append(s)
                print(f"    {code}: {s:.3f}")

            # Compute joint scaling factor
            numerator = np.sum([np.dot(root_shared_data[j], new_shared_data[j])
                               for j in range(len(merged))])
            denominator = np.sum([np.dot(new_shared_data[j], new_shared_data[j])
                                 for j in range(len(merged))])
            scale_factor = numerator / denominator
            print(f"  joint scaling factor: {scale_factor:.3f}")

        # Apply scaling to ALL conditions in new dataset
        scaled_new_data = ds_new['data'] * scale_factor

        # average shared conditions (weighted by n_subjects), add new conditions
        new_info = ds_new['info'].copy()
        new_weights = np.full(len(new_info), ds_new['n_subjects'], dtype=float)

        updated_root_data = []
        updated_root_info = []
        updated_root_weights = []

        # Process existing root conditions
        for j, row in root_info.iterrows():
            code = row['full_code']
            root_vec = root_data[j]
            root_w = root_weights[j]

            # Check if this condition exists in new dataset
            new_match = new_info[new_info['full_code'] == code]
            if len(new_match) > 0:
                new_j = new_match.index[0]
                new_vec = scaled_new_data[new_j]
                new_w = new_weights[new_j]

                # Weighted average
                combined_w = root_w + new_w
                combined_vec = (root_vec * root_w + new_vec * new_w) / combined_w

                updated_root_data.append(combined_vec)
                updated_root_weights.append(combined_w)
            else:
                updated_root_data.append(root_vec)
                updated_root_weights.append(root_w)

            updated_root_info.append(row)

        # Add new conditions not in root
        for j, row in new_info.iterrows():
            code = row['full_code']
            if code not in root_codes:
                updated_root_data.append(scaled_new_data[j])
                updated_root_info.append(row)
                updated_root_weights.append(new_weights[j])

        root_data = np.vstack(updated_root_data)
        root_info = pd.DataFrame(updated_root_info).reset_index(drop=True)
        root_weights = np.array(updated_root_weights)

        print(f"  Root now has {len(root_info)} conditions")

    # Finalize
    final_data = root_data
    final_info = root_info.copy()
    final_info['cond_id'] = pd.factorize(final_info['full_code'])[0]
    final_info['total_subjects'] = root_weights

    print(f"\n Final library: {len(final_info)} conditions ")

    return final_data, final_info


def collapse_within_dataset(datasets):
    """
    Collapse multiple sessions of the same dataset into one entry.
    since same subs, simple average of shared conditions, no weight accumulation.
    
    Args:
        datasets: list of dataset dicts with 'dataset', 'session', 'data', 'info', 'n_subjects'
    
    Returns:
        collapsed: list of dataset dicts (one per unique dataset)
    """
    # Group by dataset name (preserve order)
    grouped = OrderedDict()
    for d in datasets:
        dname = d['dataset']
        if dname not in grouped:
            grouped[dname] = []
        grouped[dname].append(d)
    
    collapsed = []
    for dname, sessions in grouped.items():
        if len(sessions) == 1:
            # Single session, just pass through
            collapsed.append(sessions[0])
            print(f"Dataset {dname}: single session, no collapse needed")
        else:
            # Multiple sessions - merge them
            print(f"Dataset {dname}: collapsing {len(sessions)} sessions")
            merged = merge_sessions(sessions)
            collapsed.append(merged)
    
    return collapsed


def merge_sessions(sessions):
    """
    Merge multiple sessions of the same dataset.
    Simple average for shared conditions (same subjects), concatenate unique ones.
    
    Args:
        sessions: list of dataset dicts from same dataset
    
    Returns:
        merged dataset dict
    """
    # Start with first session as base
    base = sessions[0]
    merged_data = {code: base['data'][i] for i, code in enumerate(base['info']['full_code'])}
    merged_counts = {code: 1 for code in base['info']['full_code']}
    merged_info = {code: base['info'].iloc[i] for i, code in enumerate(base['info']['full_code'])}
    
    session_names = [base['name']]
    
    # Merge subsequent sessions
    for sess in sessions[1:]:
        session_names.append(sess['name'])
        for i, code in enumerate(sess['info']['full_code']):
            if code in merged_data:
                # Shared condition: running average
                n = merged_counts[code]
                merged_data[code] = (merged_data[code] * n + sess['data'][i]) / (n + 1)
                merged_counts[code] = n + 1
            else:
                # New condition
                merged_data[code] = sess['data'][i]
                merged_counts[code] = 1
                merged_info[code] = sess['info'].iloc[i]
    
    # Reconstruct arrays
    codes = list(merged_data.keys())
    final_data = np.vstack([merged_data[c] for c in codes])
    final_info = pd.DataFrame([merged_info[c] for c in codes]).reset_index(drop=True)
    final_info['source'] = '+'.join(session_names)
    
    # n_subjects stays the same (same subjects across sessions)
    n_subjects = sessions[0]['n_subjects']
    
    print(f"  Collapsed {len(sessions)} sessions → {len(codes)} conditions")
    
    return {
        'dataset': sessions[0]['dataset'],
        'name': sessions[0]['dataset'],  # just dataset name now
        'data': final_data,
        'info': final_info,
        'n_subjects': n_subjects
    }

def save_library_to_cifti(data, condition_names, template_path, output_path):
    """
    Save task library to CIFTI using an existing cifti as template for brain model axis.
    
    Args:
        data: (n_conditions, P) array
        condition_names: list of condition names (length n_conditions)
        template_path: path to existing cifti to get brain model axis from, must match data cifti structure (in this case, hcp greyordinates)
        output_path: .dscalar.nii path
    """
    template = nb.load(template_path)
    
    # Get brain model axis from template (axis 1)
    bm_axis = template.header.get_axis(1)
    
    # Create new scalar axis with condition names
    scalar_axis = nb.cifti2.ScalarAxis(condition_names)
    
    # Build and save
    header = nb.Cifti2Header.from_axes((scalar_axis, bm_axis))
    cifti = nb.Cifti2Image(dataobj=data, header=header)
    nb.save(cifti, output_path)


if __name__ == "__main__":

    # define datasets for task library
    base_dir=f'{data_dir}/FunctionalFusion_new'
    sessions_to_use = {'MDTB': ['s1','s2'],
                    'Language' : ['localizer'],
                    'HCPur100' : ['task2']

                    }

    # define atlas to create library in and give the library a version
    atlas = 'fs32k'
    version = 'V1'
    data , info = make_task_library(base_dir, sessions_to_use, atlas=atlas)


    # save, use existing cifti as template for brain model axis to save the task library in  certain cifti structure
    output_dir = os.path.join(repo_dir, 'task_library')
    template_path = os.path.join(output_dir, f'template_space-{atlas}.dscalar.nii')
    output_path = os.path.join(output_dir, f'desc-tasklibrary_space-{atlas}_{version}.dscalar.nii')

    save_library_to_cifti(data=data, condition_names=info['full_code'].tolist(),
    template_path=template_path,
    output_path=output_path)
    
    path = f'{repo_dir}/task_library/desc-tasklibrary_{version}_info.tsv'
    if not os.path.exists(path):
        info.to_csv(path, sep="\t", index=False)
pass