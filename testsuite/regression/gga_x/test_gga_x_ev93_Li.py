
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ev93_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ev93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.785084672209537e+00, -1.286844622306498e+00, -5.259702703760052e-01, -1.592008331514968e-01, -9.404529436798273e-02, -1.772601304563659e-02, -3.281529986071412e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ev93_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ev93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.217960003638580e+00, -2.220190130469353e+00, -1.444206980016660e+00, -1.445662103434581e+00, -3.909022591275487e-01, -3.917526518978740e-01, -2.050562344769761e-01, -2.289392209090059e-02, -5.824700912231676e-02, -7.092864029373260e-04, -2.413731345351567e-02, -2.393292327848781e-02, -4.737445203434044e-04, -3.367859525695424e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ev93_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ev93", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.761522379737286e-04, 0.000000000000000e+00, -2.750143685558263e-04, -1.427370472808323e-03, 0.000000000000000e+00, -1.422169478116708e-03, -1.504879103379193e-01, 0.000000000000000e+00, -1.499628634660101e-01, -3.539720484341306e+00, 0.000000000000000e+00, 3.388046082621189e+00, -1.469326958811041e+02, 0.000000000000000e+00, 2.212416671155516e+01, 3.434828424878154e+00, 0.000000000000000e+00, 3.211044940525148e+00, 1.610610392744756e+01, 0.000000000000000e+00, 2.305441373851489e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
