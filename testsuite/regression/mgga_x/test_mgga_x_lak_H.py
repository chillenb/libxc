
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_lak_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.297851415477727e-01, -6.385419368494584e-01, -3.334371340207021e-01, -7.195346213573005e-02, -3.387889339262096e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_lak_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.730468624404955e-01, -4.606408212835425e-17, -8.902702986864702e-01, -2.423651340567817e-16, -5.477236136720471e-01, 1.655196581587997e-17, -9.608093684262778e-02, -3.965896441743759e-17, -4.517185852947381e-03, -3.882655250150073e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_lak_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.080370617659370e-07, 0.000000000000000e+00, 0.000000000000000e+00, -2.827454887288592e-02, 0.000000000000000e+00, 0.000000000000000e+00, -3.321211299318080e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.275452445694481e+01, 0.000000000000000e+00, 0.000000000000000e+00, 2.598003802739208e+06, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_lak_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_lak", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.787598545694576e-12, 0.000000000000000e+00, 4.968018704717751e-02, 0.000000000000000e+00, 1.239865935837565e-01, 0.000000000000000e+00, 1.687330747554389e-04, 0.000000000000000e+00, 1.271532812389318e-11, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
