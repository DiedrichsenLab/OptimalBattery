import numpy as np
import pandas as pd
import Functional_Fusion.dataset as ds
import OptimalBattery.util as ut
import nibabel as nb
import Functional_Fusion.atlas_map as am
from OptimalBattery.global_config import repo_dir
import os

def make_task_activation_library(base_dir, sessions_to_use, atlas='MNISymC3',check_scale=True, normalize=False):
      """
      Load datasets, recenter if needed, merge duplicates
      Args:
          base_dir: base directory for datasets (functional_fusion)
          sessions_to_use: dict of dataset names to list of sessions
          atlas: atlas name
          check_scale: print scale diagnostics for each dataset
          normalize: z-score within each dataset before combining

      Returns: final_data, final_info
      """
      all_data = []
      all_info = []
      scale_stats = []

      for dataset in sessions_to_use.keys():
          for sess in sessions_to_use[dataset]:
              data, info, dataset_obj = ds.get_dataset(
                  base_dir=base_dir, dataset=dataset, atlas=atlas,
                  sess=f'ses-{sess}', subj=None, type='CondAll', exclude_subjects=True
              )
              data[np.isnan(data)] = 0

              print(f'Loaded Dataset: {dataset}, session: {sess}, data shape: {data.shape}')

              # Recenter if baseline already subtracted
              if dataset_obj.subtract_baseline:
                  center_full_code = 'rest_open' if dataset == 'Nishimoto' else 'rest_task'
                  print(f'Baseline already subtracted for {dataset}, recentering')
                  data, info = ut.recenter_data(
                      data, info, center_full_code=center_full_code, keep_center=True
                  )

              # Average across subjects
              avg_data = data.mean(axis=0)

              if check_scale:
                  scale_stats.append({'dataset': f'{dataset}-{sess}', 'std': avg_data.std()})

              # Optional: normalize within dataset
              if normalize:
                  avg_data = (avg_data - avg_data.mean()) / avg_data.std()

              # Keep essential columns
              info = info[['task_code', 'cond_code']].copy()
              info['dataset_source'] = f'{dataset}-{sess}'

              all_data.append(avg_data)
              all_info.append(info)

      # Print scale diagnostics
      if check_scale:
        scale_df = pd.DataFrame(scale_stats)

        # Compute all pairwise ratios
        print('\nPairwise std ratios:')
        datasets = scale_df['dataset'].values
        stds = scale_df['std'].values
        for i in range(len(datasets)):
            for j in range(i + 1, len(datasets)):
                ratio = max(stds[i], stds[j]) / min(stds[i], stds[j])
                print(f"  {datasets[i]} vs {datasets[j]}: {ratio:.2f}x")

        std_ratio = stds.max() / stds.min()
        if std_ratio > 2:
            print(f'\n WARNING: Max ratio = {std_ratio:.2f}x, consider normalize=True')

      final_data = np.vstack(all_data)
      final_info = pd.concat(all_info, ignore_index=True)

      # Merge duplicate conditions using groupby
      final_info['_data_idx'] = range(len(final_info))

      merged_rows = []
      merged_data = []

      for (task, cond), group in final_info.groupby(['task_code', 'cond_code'], sort=False):
          indices = group['_data_idx'].values
          merged_data.append(final_data[indices].mean(axis=0))
          merged_rows.append({
              'task_code': task,
              'cond_code': cond,
              'dataset_source': group['dataset_source'].tolist()
          })

      final_data = np.vstack(merged_data)
      final_info = pd.DataFrame(merged_rows)
      final_info['cond_id'] = pd.factorize(
          final_info['task_code'] + '_' + final_info['cond_code']
      )[0]          

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
                    'HCPur100' : ['task2'],
                    'Language' : ['localizer']
                    }

    atlases = ['fs32k','SUIT3','MNISymC3']
    version = 'V1'
    for atlas_str in atlases:
        data,info = make_task_activation_library(base_dir, sessions_to_use, atlas=atlas_str, check_scale=True, normalize=False)
        conditions = info['task_code'] + '_' + info['cond_code']
        save_to_cifti(data, conditions, f'{repo_dir}/task_library/desc-tasks{version}_space-{atlas_str}_beta.dscalar.nii', atlas_str=atlas_str)
    
    path = f'{repo_dir}/task_library/desc-tasks{version}_info.tsv'
    if not os.path.exists(path):
        info.to_csv(path, sep="\t", index=False)
pass