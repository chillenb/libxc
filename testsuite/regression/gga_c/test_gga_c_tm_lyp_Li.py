
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_tm_lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tm_lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.308802401244001e-02, -4.749551825045203e-02, 2.538289223896503e-02, -2.045646231218717e-05, -7.424887660998828e-09, -3.664830560077031e-03, -5.578954301464629e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_tm_lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tm_lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.228329147688685e-02, -6.216521499588809e-02, -6.496814121497574e-02, -6.483377075081458e-02, -1.269850430144661e-01, -1.274568464570544e-01, -1.740730247901903e-05, -9.727078871252472e-02, -1.773728180439858e-09, -4.746518596066295e-02, -4.693121330924250e-03, -4.853646224172957e-03, -3.860702567899160e-05, -1.738411122782564e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_tm_lyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tm_lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [8.452780428285157e-06, 6.659490281800570e-06, 8.293466310189836e-06, 5.223845114445857e-05, 4.721926181926562e-05, 5.139849955646030e-05, 4.486919925478012e-02, 6.470339442136785e-02, 4.505294170485304e-02, -4.048481356526086e-05, 3.749489229421917e+00, 2.813127098786453e+00, 1.197569571842561e-06, 5.515242918989242e+00, 4.136432257700993e+00, 6.936017115023508e-06, 1.360573453680743e-05, 6.951057520721993e-06, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
