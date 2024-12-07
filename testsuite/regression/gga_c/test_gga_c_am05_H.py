
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_am05_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_am05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.241900194657900e-02, -2.924897905345893e-02, -2.231027840226923e-02, -1.086012927682552e-02, -1.271239885894245e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_am05_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_am05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.652614639270396e-02, -2.627593823423386e-01, -3.638768412994275e-02, -2.374069058651792e-01, -2.850879152624893e-02, -1.760431842605765e-01, -1.314663375934244e-02, -7.664546638212943e-02, -1.621037731632957e-03, -6.034382697238383e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_am05_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_am05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.106022548656963e-03, 0.000000000000000e+00, 5.607317265352121e+19, 1.613393245676680e-03, 0.000000000000000e+00, 5.378975978467906e+19, 6.694473975460594e-03, 0.000000000000000e+00, 4.350893342197411e+19, 3.226151060816446e-02, 0.000000000000000e+00, 2.292883519016902e+19, 4.887892284034644e-02, 0.000000000000000e+00, 2.711999300212808e+18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
