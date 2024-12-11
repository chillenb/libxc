
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_lsrpbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lsrpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.795474778064990e+00, -1.286434386283164e+00, -4.408493459406833e-01, -1.600524121255813e-01, -8.341539690077332e-02, -2.229214849805885e-10, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_lsrpbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lsrpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.238490037465770e+00, -2.240634101942232e+00, -1.508796725433133e+00, -1.510177894227173e+00, -4.061430215822194e-01, -4.066899147812916e-01, -2.051769125924544e-01, -2.087173952284165e-09, -7.129325380142411e-02, 1.876949526639258e-17, -1.559962115361242e-08, -6.591821675976848e-09, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_lsrpbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_lsrpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.646953852338992e-04, 0.000000000000000e+00, -2.637669174586389e-04, -1.084109762700077e-03, 0.000000000000000e+00, -1.080551246151398e-03, -8.809111407118968e-02, 0.000000000000000e+00, -8.772341821868436e-02, -4.037317497438735e+00, 0.000000000000000e+00, 1.955890267587820e-05, -8.737021599925002e+01, 0.000000000000000e+00, 0.000000000000000e+00, 1.263951812164602e-04, 0.000000000000000e+00, 5.281982833244253e-05, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
