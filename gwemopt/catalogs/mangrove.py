import io

import h5py
import numpy as np
import pandas as pd
import requests

from gwemopt.catalogs.base_catalog import BaseCatalog

URL = "https://mangrove.lal.in2p3.fr/data/mangrove.hdf5"


class MangroveCatalog(BaseCatalog):
    name = "mangrove"

    @property
    def mag_column(self) -> str:
        return "magb"

    def download_catalog(self):
        temp_path = self.get_temp_path()
        print(f"Mangrove catalog not found locally. Downloading to {temp_path}")
        get_mangrove = requests.get(URL, verify=False)
        if get_mangrove.status_code == 200:
            mangrove_bytes = io.BytesIO(get_mangrove.content)
            df = pd.DataFrame(np.array(h5py.File(mangrove_bytes)["__astropy_table__"]))
        else:
            raise ConnectionError(
                f"requests to get mangrove catalog failed with status code = {get_mangrove.status_code}\n content: {get_mangrove.content}"
            )

        key_map = {
            "RA": "ra",
            "dist": "distmpc",
            "B_mag": "magb",
            "K_mag": "magk",
            "w1mpro": "magW1",
            "w2mpro": "magW2",
            "w3mpro": "magW3",
            "w4mpro": "magW4",
            "GWGC_name": "GWGC",
            "HyperLEDA_name": "HyperLEDA",
            "2MASS_name": "2MASS",
            "SDSS-DR12_name": "SDSS",
        }

        copy_keys = ["dec", "z", "stellarmass", "PGC", "AGN_flag"]

        df = df[[x for x in key_map] + copy_keys]
        df = df.rename(columns=key_map)

        for col in ["GWGC", "HyperLEDA", "2MASS", "SDSS"]:
            df[col] = df[col].astype(str)

        df.to_hdf(self.get_catalog_path(), key="df")
        temp_path.unlink()

    def get_temp_path(self):
        return self.get_catalog_path().with_stem(f"temp_{self.name}")

    def load_catalog(self) -> pd.DataFrame:
        df = pd.read_hdf(self.get_catalog_path(), key="df")

        mask = np.where(df["distmpc"] >= 0)[0]

        return df.iloc[mask]
