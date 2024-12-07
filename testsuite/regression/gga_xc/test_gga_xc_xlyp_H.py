
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_xlyp_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_xlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.220990966746058e-01, -5.817524021815097e-01, -3.637781870927487e-01, -1.475302668221355e-01, -4.269815100018623e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_xlyp_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_xlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.281424839700010e-01, -2.361713832519182e-01, -7.172576115675632e-01, -2.499095375018286e-01, -4.029809246578913e-01, -1.966669686227086e-01, -1.148626090675543e-01, -3.555089138121431e-02, -1.066946718392586e-02, -2.453539843799461e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_xlyp_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_xlyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.029276298154162e-02, 2.924820117334720e-02, 2.192774932900085e-02, -2.713190514504380e-02, 4.646917650590301e-02, 3.480665078312204e-02, -1.792062913782291e-01, 3.986771514585075e-01, 2.989990296175629e-01, -9.212403360658151e+00, 1.702787037828197e+01, 1.277088117642037e+01, -3.696647859291039e+04, 7.157809872575426e-18, 5.368349375325281e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
