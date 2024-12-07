
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_optpbe_vdw_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optpbe_vdw", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.783513053166401e+00, -1.270025715203845e+00, -4.192699258829954e-01, -1.277565973914301e-01, -6.817997919285938e-02, -2.331228343379959e-02, -4.148479142700911e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_optpbe_vdw_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optpbe_vdw", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.254896304199213e+00, -1.529508635004909e+00, -3.568482687559550e-01, -1.620840869027109e-01, -5.845390484748567e-02, -3.101694985103884e-02, -5.531297034605314e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_optpbe_vdw_Li_restr_1_vsigma():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_optpbe_vdw", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.053488796036393e-04, -4.309659993066783e-04, -4.899287700978773e-02, -3.986747847254701e+00, -7.101405053767915e+01, -2.735190870895545e-01, -1.423649659098914e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
