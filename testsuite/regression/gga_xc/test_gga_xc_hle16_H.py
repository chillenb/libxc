
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hle16_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hle16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-8.583351191805930e-01, -7.523154811868069e-01, -4.445850188204210e-01, -2.138541908581085e-01, -1.982653172282915e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hle16_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hle16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.144446877223924e+00, -7.493481181181387e-02, -1.016382559891928e+00, -8.670738838688397e-02, -5.594071314336693e-01, -8.294319701865141e-02, -7.681714920669054e-02, -1.043391801603492e-02, -2.627620211077248e-02, 4.783201201827810e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hle16_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hle16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [5.825327403287053e-02, 0.000000000000000e+00, -3.610239927580649e+20, 7.079082992973005e-03, 0.000000000000000e+00, -2.478886955516429e+20, -6.824538415771536e-02, 0.000000000000000e+00, -9.623252876572493e+19, -2.329203354155130e+01, 0.000000000000000e+00, 4.002566251066935e+19, -7.349049479307607e+01, 0.000000000000000e+00, 2.353313560369801e+14]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
