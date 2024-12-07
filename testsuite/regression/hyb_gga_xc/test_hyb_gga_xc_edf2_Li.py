
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_edf2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_edf2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.572999523104343e+00, -1.127048229382179e+00, -3.339662147778029e-01, -1.407645737733700e-01, -6.703831071561589e-02, -8.212243709455758e-02, -3.055846285575140e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_edf2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_edf2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.991889495843997e+00, -1.993529901873505e+00, -1.383202499814904e+00, -1.384216611714868e+00, -3.807886986471442e-01, -3.809371401749072e-01, -1.815576180846825e-01, -1.123065681231491e-01, -6.894635425942995e-02, -4.504123301513960e-02, -2.931494499243554e-02, -2.946597811955034e-02, -4.391955286363386e-03, -3.865744662685552e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_edf2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_edf2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.536324839316594e-04, 3.095648845449724e-06, -1.531854776126399e-04, -5.676375307320098e-04, 2.161605430484877e-05, -5.661744928583557e-04, -4.479046713008094e-02, 2.829416040840214e-02, -4.467667737507921e-02, -2.538178178644885e+00, 2.723605500543068e+00, -7.582465723053796e+02, -4.371453065424703e+01, 1.396187854157255e+01, -2.753025197055737e+07, -6.610978097955963e+02, 4.691472320700260e-02, -6.621542553107647e+02, -8.173442895158443e+07, 0.000000000000000e+00, -2.434781172083739e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
