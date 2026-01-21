import numpy as np
import pandas as pd
import Functional_Fusion.dataset as ds
import OptimalBattery.util as ut
import nibabel as nb
import Functional_Fusion.atlas_map as am
from OptimalBattery.global_config import repo_dir
import os

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

    #Load all datasets, recenter around rest, store info and n_subs
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
                'name': f'{dataset_name}-{sess}',
                'data': avg_data,
                'info': info,
                'n_subjects': n_subjects
            })

        
    # Calibrate and merge datasets one by one, first dataset is the root
    root_data = datasets[0]['data'].copy()
    root_info = datasets[0]['info'].copy()
    root_weights = np.full(len(root_info), datasets[0]['n_subjects'], dtype=float)
    print(f"Root dataset: {datasets[0]['name']} ({datasets[0]['n_subjects']} subjects)")

    # Calibrate each subsequent dataset to root
    for i in range(1, len(datasets)):
        ds_new = datasets[i]
        ds_new['info']['full_code'] = ds_new['info']['full_code']
        print(f" Calibrating {ds_new['name']} ({ds_new['n_subjects']} subjects) ---")

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

            # Compute individual scaling factors per condition (diagnostic)
            print("  Individual scaling factors (diagnostic):")
            individual_scales = []
            for j, code in enumerate(merged['full_code'].values):
                # Scale that minimizes ||root - scale * new||^2
                # if we derive we get : scale = dot(root, new) / dot(new, new)
                s = np.dot(root_shared_data[j], new_shared_data[j]) / np.dot(new_shared_data[j], new_shared_data[j])
                individual_scales.append(s)
                print(f"    {code}: {s:.3f}")

            # Compute joint scaling factor (minimizes total RMSE across all shared conditions)
            # ||root - scale * new||^2 summed across conditions
            # Optimal: scale = sum(dot(root_i, new_i)) / sum(dot(new_i, new_i))
            numerator = np.sum([np.dot(root_shared_data[j], new_shared_data[j])
                               for j in range(len(merged))])
            denominator = np.sum([np.dot(new_shared_data[j], new_shared_data[j])
                                 for j in range(len(merged))])
            scale_factor = numerator / denominator
            print(f"  Joint scaling factor: {scale_factor:.3f}")

        # Apply scaling to ALL conditions in new dataset
        scaled_new_data = ds_new['data'] * scale_factor

        # Merge: average shared conditions (weighted by n_subjects), add new conditions
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

    # Sanity check: compare amplitude across sources
    print("\n=== Post-calibration amplitude check ===")
    for source in final_info['source'].unique():
        mask = final_info['source'] == source
        source_std = np.std(final_data[mask])
        n_conds = mask.sum()
        print(f"  {source}: std={source_std:.3f} ({n_conds} conditions)")

    print(f"\n Final library: {len(final_info)} conditions ")

    return final_data, final_info


def save_to_cifti(data,condition_names, output_path,atlas_str='fs32k'):
    """
    Save data to CIFTI using Functional Fusion atlas

    Args:
        data: (n_conditions, P) array
        condition_names: list of condition names
        output_path: .dscalar.nii path
        atlas_str: atlas name ('fs32k', 'SUIT3', etc.)
    """
    atlas, _ = am.get_atlas(atlas_str=atlas_str)
    cifti = atlas.data_to_cifti(data, row_axis=condition_names)
    nb.save(cifti, output_path)



if __name__ == "__main__":

    base_dir='Y:/data/FunctionalFusion_new'
    sessions_to_use = {'MDTB': ['s1','s2'],
                    'Language' : ['localizer'],
                    'HCPur100' : ['task2']

                    }

    atlases = ['fs32k','MNISymC3','SUIT3']
    version = 'V1'
    for atlas_str in atlases:
        data,info = make_task_library(base_dir, sessions_to_use, atlas=atlas_str)
        conditions = info['task_code'] + '_' + info['cond_code']
        save_to_cifti(data, conditions, f'{repo_dir}/task_library/desc-tasks{version}_space-{atlas_str}_beta.dscalar.nii', atlas_str=atlas_str)
    
    path = f'{repo_dir}/task_library/desc-tasks{version}_info.tsv'
    if not os.path.exists(path):
        info.to_csv(path, sep="\t", index=False)
pass