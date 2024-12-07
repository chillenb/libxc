
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_mpw91_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mpw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.803161180378499e+00, -1.291531724171503e+00, -4.269782749780079e-01, -1.605899506406449e-01, -8.102415734011781e-02, -1.712075558589937e-03, -6.560044018923878e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_mpw91_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mpw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.242102925802263e+00, -2.244221847676065e+00, -1.525222549472364e+00, -1.526581288465106e+00, -3.590552101970503e-01, -3.590394252064958e-01, -2.050341379536472e-01, -5.583617442831077e-03, -7.461534521106998e-02, -9.574001524424759e-07, -6.458634676635685e-03, -6.142903497677476e-03, -2.778352435821871e-07, -1.279996884484225e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_mpw91_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mpw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.760635203630038e-04, 0.000000000000000e+00, -2.751530136304898e-04, -1.033348995448855e-03, 0.000000000000000e+00, -1.030124434847187e-03, -1.019452432123696e-01, 0.000000000000000e+00, -1.018570120635550e-01, -4.455319750007321e+00, 0.000000000000000e+00, 3.421100710804469e+01, -7.312392015893917e+01, 0.000000000000000e+00, 4.609904277833133e+02, 3.430182947536756e+01, 0.000000000000000e+00, 3.222716405419406e+01, 3.828578645229424e+02, 0.000000000000000e+00, 5.872565727560732e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
