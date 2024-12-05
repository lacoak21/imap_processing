from typing import Optional

import h5py
import numpy as np


def load_hdf_file(path: str, num_events: Optional[int] = None) -> dict:
    """
    Loads an HDF5 file produced by the IDEX team into a dictionary where the keys
    correspond to the groups in the file, and the values are arrays containing the data.

    The HDF5 files are organized by event number, where each event contains groups
    for metadata and data items. This function consolidates all events into a single
    array to be consistent with the cdf organization.
    For example, if there are 14 events and the metadata value "category" is extracted,
    it will appear in the output dictionary as:
    {"category": np.ndarray of shape (14,)}.

    Parameters:
    ----------
    path : str
        The file path to the HDF5 file.
    num_events : int, optional
        The number of events to extract. If None, all events in the file are processed.

    Returns:
    -------
    dict
        A dictionary containing the extracted data.

    Warnings:
    --------
    - This function assumes the HDF5 file is structured with top-level groups named
      numerically (e.g., "1", "2", "3") representing events.
    """

    def collect_arrays(name, obs):
        """
        Collects data from HDF5 datasets into the dictionary `data_vars`.

        Parameters:
        ----------
        name : str
            The full path of the dataset in the HDF5 file.
        obs : h5py.Dataset or h5py.Group
            The object at the current position in the HDF5 file.
        """
        # Split the full dataset path to extract the group key and event number
        names = name.split("/")
        group_key = names[-1]
        event = names[0]

        # Check if the event name can be converted to an integer
        # This function assumes all top level groups are event numbers represented as
        # strings
        try:
            event_number = int(event)
        except ValueError as e:
            raise ValueError(
                f"Invalid group name '{event}': Top-level groups must be integers."
            ) from e

        # Process data only if the event number is within the specified range
        if event_number <= num_events:
            if isinstance(obs, h5py.Dataset):
                # Initialize storage for the group key if not already present
                if group_key not in data_vars.keys():
                    # handle arrays
                    if isinstance(obs[()], np.ndarray):
                        data_vars[group_key] = np.zeros(
                            (num_events, len(obs[()])), dtype=type(obs[()])
                        )
                    # Handle byte strings
                    elif isinstance(obs[()], bytes):
                        data_vars[group_key] = np.zeros(num_events, dtype=object)
                    # Handle other types (scalars)
                    else:
                        data_vars[group_key] = np.zeros(num_events, dtype=obs[()].dtype)
                # Convert bytes to string
                if isinstance(obs[()], bytes):
                    value = obs.asstr()[()]
                else:
                    value = obs[()]
                # store value at event number
                data_vars[group_key][int(event) - 1] = value

    # Open the HDF5 file
    f = h5py.File(path, "r")
    if not num_events or num_events > len(f.keys()):
        num_events = len(f.keys())
    # Initialize dictionary
    data_vars = {}
    # First extract metadata from each event
    f.visititems(collect_arrays)

    return data_vars
