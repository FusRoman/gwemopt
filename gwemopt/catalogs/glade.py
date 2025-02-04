import pandas as pd
from astroquery.vizier import Vizier

from gwemopt.catalogs.base_catalog import BaseCatalog

# Unset row limits when querying Vizier
Vizier.ROW_LIMIT = -1


class GladeCatalog(BaseCatalog):
    name = "glade"

    @property
    def mag_column(self) -> str:
        return "magk"

    def download_catalog(self):
        save_path = self.get_catalog_path()
        print(f"Downloading GLADE catalog to {save_path}...")
        (cat,) = Vizier.get_catalogs("VII/281/glade2", verbose=True)

        df: pd.DataFrame = cat.to_pandas()

        key_map = {
            "RAJ2000": "ra",
            "DEJ2000": "dec",
            "Dist": "distmpc",
            "Bmag": "magb",
            "Kmag": "magk",
            "2MASS": "2MASS",
        }

        copy_keys = ["GWGC", "PGC", "HyperLEDA", "z"]

        df = df[[x for x in key_map] + copy_keys]
        df.rename(columns=key_map, inplace=True)
        df["PGC"] = df["PGC"].astype(str)
        df.to_parquet(str(save_path))

    def load_catalog(self) -> pd.DataFrame:
        return pd.read_parquet(str(self.get_catalog_path()))
