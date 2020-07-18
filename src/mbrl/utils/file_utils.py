import importlib.util
import itertools
import os


def get_files_ending_with(folder_or_folders, ext):
    if isinstance(folder_or_folders, str):
        folder = folder_or_folders
        assert os.path.exists(folder)

        fnames = []
        for fname in os.listdir(folder):
            if fname.endswith(ext):
                fnames.append(os.path.join(folder, fname))
        return sorted(fnames)
    else:
        assert hasattr(folder_or_folders, '__iter__')
        return list(itertools.chain(*[get_files_ending_with(folder, ext) for folder in folder_or_folders]))


def import_config(config_fname):
    assert config_fname.endswith('.py')
    spec = importlib.util.spec_from_file_location('config', config_fname)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.params
