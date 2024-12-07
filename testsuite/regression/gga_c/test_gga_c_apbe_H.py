
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_apbe_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.198974477723475e-02, -1.675247715637752e-02, -6.897157061946935e-03, -1.157072123121813e-04, -1.660850120627222e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_apbe_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.707145616847306e-02, 1.944572352788634e+00, -4.048410805912071e-02, 8.160774889990077e+01, -2.521549536137328e-02, 4.043302375340972e+01, -6.947211714542472e-04, 2.507828273734720e-01, -1.080602966049143e-09, 7.606532686057002e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_apbe_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.937961568416993e-02, 3.875923136833987e-02, 1.937961568416993e-02, 1.064764435192871e-02, 2.129528870385742e-02, 1.064764435192871e-02, 3.968120853289963e-02, 7.936241706579927e-02, 3.968120853289963e-02, 6.510274176117674e-02, 1.302054835223529e-01, 6.510274176117674e-02, 7.077030722808280e-04, 1.415406144427693e-03, 7.077030722808280e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
