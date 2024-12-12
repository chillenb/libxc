
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_rmggac_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rmggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.381800500689693e-16, -9.584446923681654e-16, -4.021110782301765e-15, -8.633649397456134e-14, -1.955045359217807e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_rmggac_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rmggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-7.403095695405354e-17, -2.198763077912082e-01, -1.528530714365376e-16, -2.061881205505043e-01, -1.416064091412874e-15, -1.731868428931843e-01, -6.860794847304885e-14, -7.071811968127745e-02, -1.895115984783340e-11, -1.674245713528378e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rmggac_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rmggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.889115564004723e-17, 5.778231128009445e-17, 2.889115564004723e-17, 5.708739727345133e-17, 1.141747945469027e-16, 5.708739727345133e-17, 2.515459371107489e-15, 5.030918742214977e-15, 2.515459371107489e-15, 6.397721999330835e-12, 1.279544399866167e-11, 6.397721999330835e-12, 1.044523199171299e-05, 2.089046398342598e-05, 1.044523199171299e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rmggac_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rmggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-7.959730149137347e-20, -7.866414141794570e-20, -3.236756014634799e-18, -3.183993245222727e-18, -3.023870194817488e-18, -3.012883143659720e-18, -2.358864591134088e-23, -2.358803483869761e-23, -5.895407298319108e-17, -5.895407344282895e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
