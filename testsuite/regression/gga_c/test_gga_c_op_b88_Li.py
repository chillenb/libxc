
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_op_b88_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.403437709276528e-02, -4.642547239477366e-02, -1.216489322269898e-02, -1.819541205862600e-05, -3.425596884099606e-09, -1.260763397703996e-05, -4.275618939037508e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_op_b88_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-7.050790638313756e-02, -7.032916931128776e-02, -6.931761451870863e-02, -6.914291807381487e-02, -3.096755857676722e-02, -3.102259177669526e-02, -1.302754279702480e-05, -1.046179822316005e-01, -3.709349667230889e-09, -4.228443752418166e-02, -4.547089370410503e-05, -4.628321338792363e-05, -1.081763737467932e-09, -3.187629390984085e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_op_b88_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.344198087278052e-05, 0.000000000000000e+00, 2.326228895808782e-05, 1.039373432351584e-04, 0.000000000000000e+00, 1.031858274734990e-04, 8.258962142351662e-03, 0.000000000000000e+00, 8.261103383629731e-03, 5.130046488711624e-04, 0.000000000000000e+00, 6.990026144634320e+02, 2.745678723365101e-06, 0.000000000000000e+00, 2.343449931948338e+07, 2.497626218694195e-01, 0.000000000000000e+00, 2.495908656672211e-01, 1.510099787176378e+00, 0.000000000000000e+00, 1.575121740621907e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
