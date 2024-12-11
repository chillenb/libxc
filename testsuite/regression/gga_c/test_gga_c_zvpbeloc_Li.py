
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_zvpbeloc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbeloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.844408499920376e-02, -4.894241923643478e-02, -1.655043078027910e-04, -4.316078141649899e-07, -1.867041900741893e-15, -5.026608177567341e-13, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_zvpbeloc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbeloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.233846750010367e-01, -1.229653305670184e-01, -1.206404773808859e-01, -1.202597477700314e-01, -1.703512313090963e-03, -1.705927849127931e-03, -2.158160669204919e-06, -2.272726709179610e-05, -3.124227903725782e-14, -1.604764280713799e-13, -4.332871433677939e-12, -6.895195997861344e-12, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_zvpbeloc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_zvpbeloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.344006232511893e-05, 8.688012465023787e-05, 4.344006232511893e-05, 1.792480192730346e-04, 3.584960385460692e-04, 1.792480192730346e-04, 3.870037899229209e-04, 7.740075798458418e-04, 3.870037899229209e-04, 7.712394053020426e-05, 1.542478810604085e-04, 7.712394053020426e-05, 2.936261847640993e-11, 5.872523695281985e-11, 2.936261847640993e-11, 2.213724204134845e-08, 4.427448408269690e-08, 2.213724204134845e-08, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
