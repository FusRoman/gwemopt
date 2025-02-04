from pathlib import Path, PosixPath

from gwemopt.args import parse_args
from gwemopt.catalogs.get import CatalogName
from gwemopt.params import params_struct


def test_command_line():

    test_dir = Path(__file__).parent.absolute()
    test_data_dir = test_dir.joinpath("data")
    test_skymap = test_data_dir.joinpath("S190814bv_5_LALInference.v1.fits.gz")

    args = ["-o", "local_output/", "-e", str(test_skymap)]

    namespace = parse_args(args)

    params, catalog_opts, telescopes, do_3d = params_struct(namespace)

    assert len(telescopes) == 1
    assert telescopes[0].telescope_name == "ATLAS"
    assert not do_3d
    assert catalog_opts.catalog == CatalogName.NOCAT
    assert catalog_opts.catalog_dir == PosixPath("/home/roman/Data/gwemopt/catalogs")
