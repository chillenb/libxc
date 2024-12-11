
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_fd_revlb94_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_fd_revlb94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.219909624073616e-01, -5.759973562856908e-01, -3.593522531251834e-01, -2.003792045955767e-01, -9.539965557840789e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_fd_revlb94_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_fd_revlb94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.283396949160803e-01, -8.282475775863011e-17, -7.187600380559711e-01, -2.381402944146229e-16, -3.918716490370022e-01, 2.412973108158567e-17, 1.153928842526273e-02, -1.252997384552357e-16, 5.271237228335797e-01, -7.224185988981233e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_fd_revlb94_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_fd_revlb94", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.505628561735850e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.286982876023702e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.905802837174081e-01, 0.000000000000000e+00, 0.000000000000000e+00, -3.137176036538301e+01, 0.000000000000000e+00, 0.000000000000000e+00, -1.437636942651631e+06, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
