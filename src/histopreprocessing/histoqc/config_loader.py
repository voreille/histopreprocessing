import importlib.resources as pkg_resources


def load_histoqc_config(config_name="config_light.ini"):
    """
    Load a HistoQC .ini config file from the histoqc/configs folder.
    """
    try:
        # Locate the config file inside the package
        config_path = pkg_resources.files(
            "histopreprocessing.histoqc.configs") / config_name
        if not config_path.exists():
            raise FileNotFoundError(f"Config file '{config_name}' not found.")

        return config_path  # Returns a pathlib.Path-like object

    except Exception as e:
        raise RuntimeError(f"Failed to load HistoQC config: {e}")
