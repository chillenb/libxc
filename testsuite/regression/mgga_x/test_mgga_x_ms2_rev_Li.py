
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ms2_rev_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2_rev", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.890888634536117e+00, -1.244826352567415e+00, -2.816345960471011e-01, -1.732266922494205e-01, -6.085876762060433e-02, -1.713407509305537e-02, -2.969797528138526e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ms2_rev_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2_rev", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.653732753722979e+00, -2.656174366752779e+00, -1.872635386858444e+00, -1.874754684917760e+00, -3.770100086234268e-01, -3.769474282396125e-01, -2.358376919277941e-01, -2.177864853405505e-02, -8.238198954287765e-02, -6.916765007599805e-04, -2.293285346851217e-02, -2.273321948180552e-02, -4.620018047520340e-04, -2.121956314853516e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2_rev_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2_rev", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.747126972764855e-04, 0.000000000000000e+00, -4.727670218113743e-04, -2.395017163998505e-03, 0.000000000000000e+00, -2.391559175847840e-03, -2.220827613803104e-01, 0.000000000000000e+00, -2.224885149680311e-01, -5.036486012636520e+00, 0.000000000000000e+00, -1.938331164660413e-01, -1.061878849944001e+02, 0.000000000000000e+00, -1.241225139830111e+00, -8.267744164614024e-05, 0.000000000000000e+00, -1.839282898565867e-01, 2.097315329325820e-08, 0.000000000000000e+00, -2.166029103978533e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ms2_rev_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ms2_rev", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.906871619526285e-02, 1.903983075152492e-02, 3.125384554771487e-02, 3.131964213282897e-02, 2.600862452494775e-04, 2.795035259423294e-04, 1.365337787256022e-01, 9.761105659045496e-18, 1.035696342161457e-02, 1.416778486332808e-18, -1.595946310434814e-20, 3.478072782797176e-18, -2.553356255305009e-18, 3.349980847523972e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
