
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hle16_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hle16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.348140432024187e+00, -1.632672417502766e+00, -5.676044793144989e-01, -2.155710682184187e-01, -1.023598703911143e-01, -3.223150795851232e-02, -6.511387576596146e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hle16_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hle16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.167002670566760e+00, -3.169889950432494e+00, -2.160717396734547e+00, -2.162635995544342e+00, -3.082604131215058e-01, -3.082723385270379e-01, -2.938214210574636e-01, 7.196867429777997e-01, -7.449205293359917e-02, 4.840324524259733e-01, -4.445615003721067e-02, -4.318676151231184e-02, -1.227738569651230e-03, 1.330278174281523e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hle16_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hle16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([7.739199176571487e-05, 0.000000000000000e+00, 7.739158009326703e-05, -4.013396071383073e-05, 0.000000000000000e+00, -3.914436269369655e-05, -2.159886754449170e-01, 0.000000000000000e+00, -2.159490108195597e-01, 3.386222564709900e+00, 0.000000000000000e+00, 2.113080022949649e+02, -1.335721193510450e+02, 0.000000000000000e+00, 2.544169321036228e+04, -1.119549748491237e+00, 0.000000000000000e+00, -7.154530349072400e-01, -1.282460091868396e+01, 0.000000000000000e+00, 4.273776208959946e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
