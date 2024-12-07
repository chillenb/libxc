
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_htbs_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_htbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.791103110422692e+00, -1.275709954875462e+00, -4.456985849857750e-01, -1.599165593521460e-01, -7.924427667769361e-02, -2.055687026651119e-02, -3.838588870220200e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_htbs_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_htbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.254333605525768e+00, -2.256441176890871e+00, -1.543946033934467e+00, -1.545294023215452e+00, -2.918475222416654e-01, -2.929272164685991e-01, -2.057142627652462e-01, -2.615912580860030e-02, -6.032673608598020e-02, -8.296468243503852e-04, -2.750809938117428e-02, -2.730803076839329e-02, -5.541564195188991e-04, -3.939545845223386e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_htbs_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_htbs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.275450028600977e-04, 0.000000000000000e+00, -2.268030930637520e-04, -8.235545095560432e-04, 0.000000000000000e+00, -8.210601756689223e-04, -1.466126496710741e-01, 0.000000000000000e+00, -1.459977104202936e-01, -3.683450477528793e+00, 0.000000000000000e+00, 0.000000000000000e+00, -9.919766472408546e+01, 0.000000000000000e+00, 0.000000000000000e+00, -6.632227216413753e-309, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
