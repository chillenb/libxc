
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_tau_hcth_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.878691494614327e-01, -5.992384465617840e-01, -3.486849452199785e-01, -1.955739593523950e-01, -1.577627680623627e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_tau_hcth_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.184285807450800e-01, -3.370435348037611e-17, -8.269840647577708e-01, -2.897835227507471e-16, -4.465296562491800e-01, -1.014429770569856e-17, -4.146545038071896e-02, -1.042822104356203e-16, -2.099642347520668e-02, -1.098286664931695e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tau_hcth_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.948663742879907e-02, 0.000000000000000e+00, 0.000000000000000e+00, 1.300912117264313e-02, 0.000000000000000e+00, 0.000000000000000e+00, -7.330650184564187e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.067317578864333e+01, 0.000000000000000e+00, 0.000000000000000e+00, -3.083715157524038e+01, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_tau_hcth_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_tau_hcth", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-4.146678820152912e-05, 0.000000000000000e+00, -1.004024219698815e-05, 0.000000000000000e+00, 1.825123978578562e-02, 0.000000000000000e+00, -4.205265084907789e-02, 0.000000000000000e+00, -1.892648521086459e-08, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
