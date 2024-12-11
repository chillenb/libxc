
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_hle17_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_hle17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.803442361635423e-01, -7.178839861819475e-01, -4.201730545869505e-01, -1.189222347822500e-01, -6.090831489520344e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_hle17_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_hle17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.068770766942780e+00, -3.206406185149444e-02, -9.608555702211845e-01, -1.253000864034932e-01, -5.598196860992748e-01, -9.755903163384445e-02, -1.569119568143394e-01, -4.525865555655153e-02, -8.075376490203907e-03, -3.535639951086714e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_hle17_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_hle17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.292275478943461e+00, 2.623015059558103e+00, 1.311507529779051e+00, -2.500095544116812e-02, 2.611078398318575e-02, 1.305539199159287e-02, -5.770760482808567e-02, 2.231029207236555e-01, 1.115514603618278e-01, 1.745462325057117e+01, 4.386183810518461e+01, 2.193091905259231e+01, 3.956728219476338e+06, 8.904666062062582e+06, 4.452333031031291e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_hle17_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_hle17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.023208673577882e+01, -3.091254991392247e+00, 8.842101577129651e-03, -8.679116293740834e-80, 2.425875769498348e-03, -3.688449287420135e-77, -7.800670582774917e-04, -2.243592858215047e-59, -3.233640786383449e-08, -3.632322626126131e-44])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
