
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_gea2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.141325803425452e-03, 0.000000000000000e+00, 2.135548041553571e-03, 6.398157552748461e-03, 0.000000000000000e+00, 6.381848088126381e-03, 4.614896372377461e-01, 0.000000000000000e+00, 4.620789138692905e-01, 2.894480549942088e+00, 0.000000000000000e+00, 8.699790635302610e+03, 4.646130896403081e+01, 0.000000000000000e+00, 2.727085252418318e+08, 7.481638617199318e+03, 0.000000000000000e+00, 7.647285960080888e+03, 9.151270127775289e+08, 0.000000000000000e+00, 2.547064861199871e+09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_gea2_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [1.666666666666667e-01, 1.666666666666666e-01, 1.666666666666666e-01, 1.666666666666665e-01, 1.666666666666667e-01, 1.666666666666667e-01, 1.666666666666666e-01, 1.666666666666298e-01, 1.666666666666667e-01, 1.666666665320156e-01, 1.666666666666667e-01, 1.666666666666668e-01, 1.666666666666665e-01, 1.666666666666666e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_gea2_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_gea2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
