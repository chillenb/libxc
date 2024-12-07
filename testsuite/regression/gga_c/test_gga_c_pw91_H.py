
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pw91_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.201512807830702e-02, -1.902427530777540e-02, -9.006338783824008e-03, -1.828666574793959e-04, -2.361794657415595e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pw91_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.699613509936388e-02, 1.690083436176813e+00, -4.025609054887966e-02, 7.673802215393304e+01, -2.840244884935649e-02, 4.810944149310160e+01, -1.084064256567135e-03, 4.491607433521928e-01, -1.537559180236209e-09, 1.109794380005358e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pw91_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.774967531882661e-02, 3.549935063765323e-02, 1.774967531882661e-02, 9.246684718035291e-03, 1.849336943607058e-02, 9.246684718035291e-03, 4.118620431803204e-02, 8.237240863606407e-02, 4.118620431803204e-02, 1.005551859654972e-01, 2.011103719309944e-01, 1.005551859654972e-01, 1.006348196414432e-03, 2.012696392828865e-03, 1.006348196414432e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
