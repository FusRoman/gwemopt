import tempfile
from pathlib import Path

from pytest import mark

from gwemopt.catalogs.get import CatalogOpts, get_catalog
from gwemopt.catalogs.glade import GladeCatalog
from gwemopt.catalogs.mangrove import MangroveCatalog
from gwemopt.catalogs.nedlvs import NEDCatalog
from gwemopt.catalogs.twomrs import TwoMRSCatalog
from gwemopt.io.skymap import read_skymap


def test_mangrove(skymap_path: Path):
    galaxy_limit = 100

    with tempfile.TemporaryDirectory() as temp_dir:
        cat_opts = CatalogOpts("MANGROVE", "Smass", galaxy_limit, Path(temp_dir))
        cat = cat_opts.catalog_from_str()

        assert isinstance(cat, MangroveCatalog)

        map_struct, _ = read_skymap(
            skymap_path=skymap_path,
            nside_raster=512,
            galactic_limit=0.0,
            confidence_level=1.0,
        )

        map_struct, cat_df = get_catalog(
            cat_opts, 0.5, 1.0, 512, map_struct, Path(temp_dir)
        )

        assert len(cat_df) == galaxy_limit
        assert list(cat_df.columns) == [
            "ra",
            "distmpc",
            "magb",
            "magk",
            "magW1",
            "magW2",
            "magW3",
            "magW4",
            "GWGC",
            "HyperLEDA",
            "2MASS",
            "SDSS",
            "dec",
            "z",
            "stellarmass",
            "PGC",
            "AGN_flag",
            "grade",
            "S",
            "Sloc",
            "Smass",
        ]


def test_2mrs(skymap_path: Path):
    galaxy_limit = 100

    with tempfile.TemporaryDirectory() as temp_dir:
        cat_opts = CatalogOpts("2MRS", "S", galaxy_limit, Path(temp_dir))
        cat = cat_opts.catalog_from_str()

        assert isinstance(cat, TwoMRSCatalog)

        map_struct, _ = read_skymap(
            skymap_path=skymap_path,
            nside_raster=512,
            galactic_limit=0.0,
            confidence_level=1.0,
        )

        map_struct, cat_df = get_catalog(
            cat_opts, 0.5, 1.0, 512, map_struct, Path(temp_dir)
        )

        assert len(cat_df) == galaxy_limit
        assert list(cat_df.columns) == [
            "ra",
            "dec",
            "magk",
            "z",
            "distmpc",
            "grade",
            "S",
            "Sloc",
            "Smass",
        ]


@mark.skip(reason="Catalog too large")
def test_ned(skymap_path: Path):
    galaxy_limit = 100

    with tempfile.TemporaryDirectory() as temp_dir:
        cat_opts = CatalogOpts("NED", "Sloc", galaxy_limit, Path(temp_dir))
        cat = cat_opts.catalog_from_str()

        assert isinstance(cat, NEDCatalog)

        map_struct, _ = read_skymap(
            skymap_path=skymap_path,
            nside_raster=512,
            galactic_limit=0.0,
            confidence_level=1.0,
        )

        map_struct, cat_df = get_catalog(
            cat_opts, 0.5, 1.0, 512, map_struct, Path(temp_dir)
        )

        assert len(cat_df) == galaxy_limit
        assert list(cat_df.columns) == [
            "name",
            "ra",
            "dec",
            "objtype",
            "redshift",
            "redshift_error",
            "z_tech",
            "z_qual",
            "z_qual_flag",
            "z_refcode",
            "ziDist",
            "ziDist_unc",
            "ziDist_method",
            "ziDist_indicator",
            "ziDist_refcode",
            "distmpc",
            "distmpc_unc",
            "DistMpc_method",
            "ebv",
            "A_FUV_MWext",
            "A_NUV_MWext",
            "A_J_MWext",
            "A_H_MWext",
            "A_Ks_MWext",
            "A_W1_MWext",
            "A_W2_MWext",
            "A_W3_MWext",
            "A_W4_MWext",
            "mag_fuv",
            "m_FUV_unc",
            "mag_nuv",
            "m_NUV_unc",
            "Lum_FUV",
            "Lum_FUV_unc",
            "Lum_NUV",
            "Lum_NUV_unc",
            "GALEXphot",
            "m_J",
            "m_J_unc",
            "m_H",
            "m_H_unc",
            "magk",
            "m_Ks_unc",
            "Lum_J",
            "Lum_J_unc",
            "Lum_H",
            "Lum_H_unc",
            "Lum_Ks",
            "Lum_Ks_unc",
            "tMASSphot",
            "m_W1",
            "m_W1_unc",
            "m_W2",
            "m_W2_unc",
            "m_W3",
            "m_W3_unc",
            "m_W4",
            "m_W4_unc",
            "Lum_W1",
            "Lum_W1_unc",
            "Lum_W2",
            "Lum_W2_unc",
            "Lum_W3",
            "Lum_W3_unc",
            "Lum_W4",
            "Lum_W4_unc",
            "WISEphot",
            "sfr_w4",
            "SFR_W4_unc",
            "SFR_hybrid",
            "SFR_hybrid_unc",
            "ET_flag",
            "mstar",
            "Mstar_unc",
            "MLratio",
            "grade",
            "S",
            "Sloc",
            "Smass",
        ]


def test_glade(skymap_path: Path):
    galaxy_limit = 100

    with tempfile.TemporaryDirectory() as temp_dir:
        cat_opts = CatalogOpts("GLADE", "Sloc", galaxy_limit, Path(temp_dir))
        cat = cat_opts.catalog_from_str()

        assert isinstance(cat, GladeCatalog)

        map_struct, _ = read_skymap(
            skymap_path=skymap_path,
            nside_raster=512,
            galactic_limit=0.0,
            confidence_level=1.0,
        )

        map_struct, cat_df = get_catalog(
            cat_opts, 0.5, 1.0, 512, map_struct, Path(temp_dir)
        )

        assert len(cat_df) == galaxy_limit
        assert list(cat_df.columns) == [
            "ra",
            "dec",
            "distmpc",
            "magb",
            "magk",
            "2MASS",
            "GWGC",
            "PGC",
            "HyperLEDA",
            "z",
            "grade",
            "S",
            "Sloc",
            "Smass",
        ]
