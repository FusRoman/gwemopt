import copy
from enum import Enum
from pathlib import Path

import healpy as hp
import numpy as np
from astroquery.vizier import Vizier
from ligo.skymap.bayestar import derasterize
from scipy.stats import norm

from gwemopt.catalogs.base_catalog import BaseCatalog
from gwemopt.catalogs.clu import CluCatalog
from gwemopt.catalogs.glade import GladeCatalog
from gwemopt.catalogs.mangrove import MangroveCatalog
from gwemopt.catalogs.nedlvs import NEDCatalog
from gwemopt.catalogs.twomrs import TwoMRSCatalog

# Unset row limits when querying Vizier
Vizier.ROW_LIMIT = -1


class CatalogName(Enum):
    TWOMRS = "2MRS"
    GLADE = "GLADE"
    CLU = "CLU"
    MANGROVE = "MANGROVE"
    NED = "NED"
    NOCAT = "NOCATALOG"


class GalGrade(Enum):
    S = "S"
    Sloc = "Sloc"
    Smass = "Smass"


class CatalogOpts:

    def __init__(
        self, catalog_name: str, galaxy_grade: str, galaxy_limit: int, catalog_dir: Path
    ) -> None:
        catalog_name = "TWOMRS" if catalog_name == "2MRS" else catalog_name
        self.catalog = CatalogName[catalog_name]
        self.galaxy_grade = GalGrade[galaxy_grade]
        if self.catalog != CatalogName.MANGROVE and self.galaxy_grade == GalGrade.Smass:
            raise ValueError(
                "You are trying to use the stellar mass information (Smass), "
                "please select the mangrove catalog for such use."
            )
        self.galaxy_limit = galaxy_limit
        self.catalog_dir = catalog_dir
        self.catalog_dir.mkdir(parents=True, exist_ok=True)

    def catalog_from_str(self) -> BaseCatalog:
        match self.catalog:
            case CatalogName.TWOMRS:
                return TwoMRSCatalog(catalog_dir=self.catalog_dir)
            case CatalogName.GLADE:
                return GladeCatalog(catalog_dir=self.catalog_dir)
            case CatalogName.MANGROVE:
                return MangroveCatalog(catalog_dir=self.catalog_dir)
            case CatalogName.NED:
                return NEDCatalog(catalog_dir=self.catalog_dir)
            case CatalogName.CLU:
                return CluCatalog(catalog_dir=self.catalog_dir)


def get_catalog(
    catalog_opts: CatalogOpts,
    confidence_level: float,
    powerlaw_dist_exp: float,
    nside: int,
    map_struct,
    output_dir: Path,
    export_catalog: bool = True,
):
    """
    Get the catalog of galaxies to be used in the optimization.
    """

    """AB Magnitude zero point."""
    MAB0 = -2.5 * np.log10(3631.0e-23)
    pc_cm = 3.08568025e18
    const = 4.0 * np.pi * (10.0 * pc_cm) ** 2.0

    cat = catalog_opts.catalog_from_str()

    cat_df = cat.get_catalog()

    if catalog_opts.catalog == CatalogName.GLADE:
        # Keep only galaxies with finite B mag when using it in the grade
        if catalog_opts.galaxy_grade == GalGrade.S:
            mask = np.where(~np.isnan(cat_df["magb"]))[0]
            cat_df = cat_df.iloc[mask]

    prob_scaled = copy.deepcopy(map_struct["skymap_raster_schedule"]["PROB"])
    prob_sorted = np.sort(prob_scaled)[::-1]
    prob_indexes = np.argsort(prob_scaled)[::-1]
    prob_cumsum = np.cumsum(prob_sorted)
    index = np.argmin(np.abs(prob_cumsum - confidence_level)) + 1
    prob_scaled[prob_indexes[index:]] = 0.0

    ipix = hp.ang2pix(
        nside,
        np.array(cat_df["ra"]),
        np.array(cat_df["dec"]),
        lonlat=True,
    )

    if "DISTNORM" in map_struct:
        if map_struct["skymap_raster_schedule"]["DISTNORM"] is not None:
            # creat an mask to cut at 3 sigma in distance
            mask = np.zeros(len(cat_df["distmpc"]))

            condition_indexer = np.where(
                (
                    cat_df["distmpc"]
                    < (
                        map_struct["skymap_raster_schedule"]["DISTMEAN"][ipix]
                        + (3 * map_struct["skymap_raster_schedule"]["DISTSTD"][ipix])
                    )
                )
                & (
                    cat_df["distmpc"]
                    > (
                        map_struct["skymap_raster_schedule"]["DISTMEAN"][ipix]
                        - (3 * map_struct["skymap_raster_schedule"]["DISTSTD"][ipix])
                    )
                )
            )
            mask[condition_indexer] = 1

            s_loc = (
                prob_scaled[ipix]
                * (
                    map_struct["skymap_raster_schedule"]["DISTNORM"][ipix]
                    * norm(
                        map_struct["skymap_raster_schedule"]["DISTMU"][ipix],
                        map_struct["skymap_raster_schedule"]["DISTSIGMA"][ipix],
                    ).pdf(cat_df["distmpc"])
                )
                ** powerlaw_dist_exp
                / map_struct["pixarea"]
            )

            # multiplie the Sloc by 1 or 0 according to the 3 sigma condistion
            s_loc = s_loc * mask
        else:
            s_loc = copy.copy(prob_scaled[ipix])
    else:
        s_loc = copy.copy(prob_scaled[ipix])

    # this happens when we are using a tiny catalog...
    if np.all(s_loc == 0.0):
        s_loc[:] = 1.0

    # new version of the Slum calcul (from HOGWARTs)
    Lsun = 3.828e26
    Msun = 4.83
    Lblist = []

    for _, row in cat_df.iterrows():
        if row[cat.mag_column] is not None:
            Mb = row[cat.mag_column] - 5 * np.log10((row["distmpc"] * 10**6)) + 5
            Lb = Lsun * 2.512 ** (Msun - Mb)
            Lblist.append(Lb)
        else:
            Lblist.append(0)

    # set 0 when Sloc is 0 (keep compatible galaxies for normalization)
    Lblist = np.array(Lblist)

    Lblist[s_loc == 0] = 0

    Slum = Lblist / np.nansum(np.array(Lblist))

    mlim, M_KNmin, M_KNmax = 22, -17, -12
    L_KNmin = const * 10.0 ** ((M_KNmin + MAB0) / (-2.5))
    L_KNmax = const * 10.0 ** ((M_KNmax + MAB0) / (-2.5))

    Llim = (
        4.0
        * np.pi
        * (cat_df["distmpc"] * 1e6 * pc_cm) ** 2.0
        * 10.0 ** ((mlim + MAB0) / (-2.5))
    )
    sdet = (L_KNmax - Llim) / (L_KNmax - L_KNmin)
    sdet[sdet < 0.01] = 0.01
    sdet[sdet > 1.0] = 1.0

    # Set nan values to zero
    s_loc[np.isnan(s_loc)] = 0
    Slum[np.isnan(Slum)] = 0

    if (
        catalog_opts.catalog == CatalogName.MANGROVE
        and catalog_opts.galaxy_grade == GalGrade.Smass
    ):
        print("Use of the stellar mass information (Smass) from the MANGROVE catalog")
        # set Smass
        s_mass = cat_df["stellarmass"].to_numpy()

        # put null values to nan
        s_mass[np.where(s_mass == 0)] = np.nan

        # go back to linear scaling and not log
        s_mass = 10.0**s_mass

        # Keep only galaxies with finite stellarmass when using it in the grade
        # set nan values to 0
        s_mass[~np.isfinite(s_mass)] = 0

        # set Smass
        s_mass = s_mass / np.sum(s_mass)

        # alpha is defined only with non null mass galaxies, we set a mask for that
        ind_without_mass = np.where(s_mass == 0)
        Sloc_temp = copy.deepcopy(s_loc)
        Sloc_temp[ind_without_mass] = 0

        # alpha_mass parameter is defined in such way that in mean Sloc count
        # in as much as Sloc*alpha*Smass
        alpha_mass = np.sum(Sloc_temp) / np.sum(Sloc_temp * s_mass)
        print(
            "You chose to use the grade using stellar mass, the parameters values are:"
        )
        print("alpha_mass =", alpha_mass)

        # beta_mass is a parameter allowing to change the importance of Sloc
        # according to Sloc*alpha*Smass
        # beta_mass should be fitted in the futur on real GW event
        # for which we have the host galaxy
        # fixed to one at the moment
        beta_mass = 1
        print("beta_mass =", beta_mass)

        s_mass = s_loc * (1 + (alpha_mass * beta_mass * s_mass))

    s = np.array(s_loc * Slum * sdet)
    prob = np.zeros(map_struct["skymap_raster_schedule"]["PROB"].shape)
    match catalog_opts.galaxy_grade:
        case GalGrade.Sloc:
            for j in range(len(ipix)):
                prob[ipix[j]] += s_loc[j]
            grade = s_loc
        case GalGrade.S:
            for j in range(len(ipix)):
                prob[ipix[j]] += s[j]
            grade = s
        case GalGrade.Smass:
            for j in range(len(ipix)):
                prob[ipix[j]] += s_mass[j]
            grade = s_mass

    prob[np.isnan(prob)] = 0.0
    prob = prob / np.sum(prob)

    map_struct["skymap_raster_schedule"]["PROB"] = prob
    skymap = map_struct["skymap_raster_schedule"].copy()
    map_struct["skymap"] = derasterize(skymap)

    cat_df["grade"] = grade
    cat_df["S"] = s
    cat_df["Sloc"] = s_loc
    if catalog_opts.galaxy_grade != GalGrade.Smass:
        cat_df["Smass"] = 1.0
    else:
        cat_df["Smass"] = s_mass

    mask = np.where(~np.isnan(grade))[0]
    cat_df = cat_df.iloc[mask]

    cat_df.sort_values(by=["grade"], inplace=True, ascending=False, ignore_index=True)

    if len(cat_df) > catalog_opts.galaxy_limit:
        print(f"Cutting catalog to top {catalog_opts.galaxy_limit} galaxies...")
        cat_df = cat_df.iloc[: catalog_opts.galaxy_limit]

    # now normalize the distributions
    for key in ["S", "Sloc", "Smass"]:
        cat_df[key] = cat_df[key] / np.sum(cat_df[key])

    if export_catalog:
        output_path = output_dir.joinpath(f"catalog_{cat.name}.csv")
        print(f"Saving catalog to {output_path}")
        cat_df.to_csv(output_path)

    return map_struct, cat_df
