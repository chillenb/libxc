
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_cf22d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.660150188349701e-01, -5.284555702975646e-01, -3.352494587187449e-01, -4.063509334037751e-02, -5.458037440319759e-02, 7.894262372270689e-04, -6.261875434402867e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_cf22d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.124321082978714e+00, 2.134896928985225e+00, -1.521435732120051e-01, -1.496286757547687e-01, -2.658831939633818e-01, -2.635584367436056e-01, -1.408756698852953e-01, 1.049417927822357e-03, -3.766805754618382e-02, -5.723597749556967e-07, 1.314826628999736e-03, 1.129862809845112e-03, -7.274915826758866e-07, -6.653385800300909e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_cf22d_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.817376829657021e-03, 0.000000000000000e+00, -1.817134948997509e-03, -1.550701484749001e-03, 0.000000000000000e+00, -1.557355419884665e-03, -1.963697598248751e-01, 0.000000000000000e+00, -1.992717087463527e-01, 1.090619403729775e+00, 0.000000000000000e+00, 1.508685241681361e-01, -1.575920491536713e+02, 0.000000000000000e+00, 1.343307556864727e+00, 1.558487836977798e-01, 0.000000000000000e+00, 1.405015806623695e-01, 9.807046531341311e-01, 0.000000000000000e+00, 1.406068112304475e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_cf22d_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_cf22d_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.577206854532988e-01, -1.583382410911967e-01, -3.175994911031681e-02, -3.178988809827522e-02, 3.117332434859516e-02, 3.220047494606236e-02, 2.245237544540498e+00, 1.695440705054323e-05, 3.049117091206580e-01, 3.352301803803427e-09, 8.394143191573483e-09, 1.832587899051539e-05, 4.558732858219504e-20, 3.737581550405177e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
