
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_task_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.297848168767422e-01, -6.540564673681950e-01, -3.815550537367497e-01, -9.699639630712219e-02, -2.171148337221141e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_task_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.717022858340814e-01, -1.239380978408563e-16, -8.024355533418041e-01, -2.217421765178426e-16, -3.743746739077583e-01, 3.932375424058545e-17, -1.396732959257623e-01, -4.244900632553053e-17, 4.920180302939378e-03, 1.848135317825179e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_task_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.497967366567078e-02, 0.000000000000000e+00, 0.000000000000000e+00, -8.745688305286424e-02, 0.000000000000000e+00, 0.000000000000000e+00, -7.990386976365116e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.164406541511827e+00, 0.000000000000000e+00, 0.000000000000000e+00, -1.806017046859976e+04, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_task_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.311259508117765e-01, 0.000000000000000e+00, 1.515998548604367e-01, 0.000000000000000e+00, 2.782957035261471e-01, 0.000000000000000e+00, 9.451802457229562e-11, 0.000000000000000e+00, 1.295145269122787e-02, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
