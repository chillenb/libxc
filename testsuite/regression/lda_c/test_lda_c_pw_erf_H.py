
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_pw_erf_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw_erf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.299136039416019e-02, -2.113328129863592e-02, -1.165532639407088e-02, -2.500068150490756e-04, -9.720218380193657e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_pw_erf_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_pw_erf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.862995778266198e-02, 1.150123207541889e+02, -2.685272814785892e-02, 9.883664830139492e+01, -1.750889243539751e-02, 3.061843339318472e+01, -5.941473965577031e-04, 1.306963624444586e-02, -2.572485880997927e-10, -1.076149054720634e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
