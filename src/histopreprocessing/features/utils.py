import json

import h5py


def save_hdf5(output_path, asset_dict, global_attr_dict=None, mode="a"):
    """
    Save data to an HDF5 file.

    Parameters:
        output_path (str): Path to the HDF5 file.
        asset_dict (dict): Dictionary of datasets to save.
        global_attr_dict (dict): Dictionary of global attributes for the WSI.
        mode (str): File mode ('w' for write, 'a' for append).
    """
    file = h5py.File(output_path, mode)

    # Add global attributes to the file
    if global_attr_dict is not None:
        for attr_key, attr_val in global_attr_dict.items():
            # Serialize unsupported types to JSON strings
            if isinstance(attr_val, (dict, list)):
                attr_val = json.dumps(attr_val)
            file.attrs[attr_key] = attr_val

    # Add datasets and their attributes
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1,) + data_shape[1:]
            maxshape = (None,) + data_shape[1:]
            dset = file.create_dataset(
                key,
                shape=data_shape,
                maxshape=maxshape,
                chunks=chunk_shape,
                dtype=data_type,
            )
            dset[:] = val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0] :] = val

    file.close()
    return output_path


def load_hdf5(input_path):
    """
    Load data from an HDF5 file.

    Parameters:
        input_path (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary containing the embeddings, coordinates, and global attributes.
    """
    result = {}

    with h5py.File(input_path, "r") as file:
        # Load global attributes
        global_attrs = {}
        for attr_key, attr_val in file.attrs.items():
            # Deserialize JSON strings if necessary
            if isinstance(attr_val, str) and attr_val.startswith("{"):
                try:
                    attr_val = json.loads(attr_val)
                except json.JSONDecodeError:
                    pass
            global_attrs[attr_key] = attr_val
        result["global_attributes"] = global_attrs

        # Load datasets
        if "embeddings" in file:
            result["embeddings"] = file["embeddings"][:]
        else:
            raise KeyError("Dataset 'embeddings' not found in the HDF5 file.")

        if "coordinates" in file:
            result["coordinates"] = file["coordinates"][:]
        else:
            raise KeyError("Dataset 'coordinates' not found in the HDF5 file.")

    return result
