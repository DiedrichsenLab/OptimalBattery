import Functional_Fusion.dataset as ds

base_dir = 'Y:/data/FunctionalFusion_new'

sessions_to_use = {
    'MDTB': ['s1', 's2'],
    'Language': ['localizer'],
    'HCPur100': ['task2']
}

sessions_to_use = {
    'HCPur100': ['task2']
}

atlases = [ 'MNIAsymSubcortical']

for dataset_name, sessions in sessions_to_use.items():
    dataset_obj = ds.get_dataset_class(base_dir, dataset_name)
    for sess in sessions:
        for atlas in atlases:
            print(f"Extracting {dataset_name} ses-{sess} to {atlas}")
            dataset_obj.extract_all(ses_id=f'ses-{sess}', type='CondAll', atlas=atlas, smooth=None)