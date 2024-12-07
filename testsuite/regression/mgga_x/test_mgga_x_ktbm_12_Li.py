
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_12_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.101487583327224e+00, -1.470627022278618e+00, -3.098166073139247e-01, -1.884867435871463e-01, -7.019917716560013e-02, -1.140257015529293e-02, -2.110937628863581e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_12_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.609837122488210e+00, -2.612359882841651e+00, -1.751931134793646e+00, -1.753347994420102e+00, -3.873880370040303e-01, -3.887756040433612e-01, -2.410126388493669e-01, -1.395175139242141e-02, -8.718524496660630e-02, -4.424650671962358e-04, -1.503286382451870e-02, -1.456455132054160e-02, -3.028682649114963e-04, -2.101028082892448e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_12_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.770390257162032e-04, 0.000000000000000e+00, -6.747394188452645e-04, -2.662153429052193e-03, 0.000000000000000e+00, -2.654925638973987e-03, -6.483656336561534e-02, 0.000000000000000e+00, -6.811969181418935e-02, -1.035160360319204e+01, 0.000000000000000e+00, -2.006264816296378e+01, -9.176820735811650e+01, 0.000000000000000e+00, -5.025023254564568e+04, 6.586353384667024e-02, 0.000000000000000e+00, -1.793836565628586e+01, 1.554380709890800e-01, 0.000000000000000e+00, -2.274906425213554e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_12_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_12_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_12", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.903974852063355e-02, 2.901149730973476e-02, 4.347609463144557e-02, 4.345718335587922e-02, 1.981990622748859e-02, 2.146667427062580e-02, 3.098822996389044e-01, 2.565295809146883e-04, 2.974368036815194e-01, 2.047378390438201e-05, -2.021420492610777e-08, 2.609542124052305e-04, -1.494418517418669e-16, 9.923873481888576e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
