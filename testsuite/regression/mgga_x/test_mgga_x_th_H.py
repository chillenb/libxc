
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_th_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_th", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.553235016247684e+02, -1.941049502208377e+00, -3.654026788929930e-01, -6.831937031059896e-03, -5.302203101210946e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_th_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_th", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.339921368927367e+02, 4.848646408333689e-14, -5.823148506625159e+00, -8.946227413571680e-16, -1.096208036679004e+00, -2.196718776887313e-16, -2.049581109320471e-02, -2.485192424038757e-18, -1.590660948937704e-06, 2.030275451787791e-23])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_th_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_th", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.308017340842908e+03, 0.000000000000000e+00, 0.000000000000000e+00, -6.233269644380602e-01, 0.000000000000000e+00, 0.000000000000000e+00, -5.517141994537036e-01, 0.000000000000000e+00, 0.000000000000000e+00, -5.316560594048571e-01, 0.000000000000000e+00, 0.000000000000000e+00, -2.929197613394972e-01, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_th_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_th", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.827195000271280e+04, 0.000000000000000e+00, 4.133620510251133e+00, 0.000000000000000e+00, 7.320906204885298e-01, 0.000000000000000e+00, 1.343678208257808e-02, 0.000000000000000e+00, 7.740459368115901e-07, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
