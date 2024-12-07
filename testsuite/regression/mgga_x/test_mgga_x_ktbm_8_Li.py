
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_8_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_8", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.060239690920034e+00, -1.445260955596779e+00, -3.026448857537621e-01, -1.846432239002755e-01, -6.902554131431997e-02, -1.076135525585338e-02, -1.959368308380410e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_8_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_8", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.566791456845050e+00, -2.569252110812215e+00, -1.729946508185770e+00, -1.731330863422644e+00, -3.704460848451479e-01, -3.717879038023114e-01, -2.364864045575713e-01, -1.288407596726987e-02, -8.529762453706792e-02, -4.085827658298375e-04, -1.379481837703954e-02, -1.345002560169517e-02, -2.778565400461032e-04, -1.940139093180831e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_8_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_8", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.853151361807215e-04, 0.000000000000000e+00, -5.832948932686252e-04, -2.356775227412514e-03, 0.000000000000000e+00, -2.350095397656263e-03, -7.834018597678825e-02, 0.000000000000000e+00, -8.198399515582289e-02, -8.869695012422161e+00, 0.000000000000000e+00, -3.577994072713658e+01, -9.798202767348172e+01, 0.000000000000000e+00, -8.973970133373802e+04, -1.685274971746779e-01, 0.000000000000000e+00, -3.198876769915625e+01, -3.209685856006144e-01, 0.000000000000000e+00, -4.062667527048872e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_8_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_8", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_8_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_8", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.312234554891465e-02, 2.309608900148049e-02, 3.672620193026718e-02, 3.669899263441108e-02, 2.357878307888943e-02, 2.537413313920213e-02, 2.403082236391434e-01, 4.573051837164148e-04, 3.185105607779767e-01, 3.656319950840653e-05, 5.104425388414349e-08, 4.651414712842967e-04, 3.085834670139392e-16, 1.772265800306036e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
