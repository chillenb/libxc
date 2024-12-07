
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_mpw91_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mpw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.219293156395452e-01, -5.804375630705180e-01, -3.616264121646542e-01, -1.427493237453912e-01, -3.830703871922482e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_mpw91_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mpw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.282527693597577e-01, -3.020059758765095e-17, -7.187790747749849e-01, -1.525828560482436e-16, -4.051473736058350e-01, -4.145321881809923e-18, -1.179929320113321e-01, -7.237157848204055e-17, -1.411331019009875e-03, -6.953494089453197e-20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_mpw91_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mpw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.512883559822907e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.561089479820033e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.682091327286182e-01, 0.000000000000000e+00, 0.000000000000000e+00, -8.142530715346004e+00, 0.000000000000000e+00, 0.000000000000000e+00, 7.196262787583827e+02, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
