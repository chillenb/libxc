
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_mpw1pw_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1pw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.415225131629273e+00, -1.015734571427618e+00, -3.242242550754696e-01, -1.356623103917342e-01, -6.221998312263116e-02, -1.284064521107632e-03, -4.920033032057850e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_mpw1pw_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1pw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.794271191489143e+00, -1.795732664968025e+00, -1.243148246510435e+00, -1.244069696411814e+00, -2.890949093743341e-01, -2.890901948805904e-01, -1.778368398900809e-01, -1.012135420897922e-01, -6.292590366294043e-02, 4.054853320756937e-01, -4.844026897713779e-03, -4.607228768639580e-03, -2.083764338227555e-07, -9.599976768002959e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_mpw1pw_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_mpw1pw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.672110095672999e-04, 7.967326140990596e-05, -1.665281295179144e-04, -6.442409345772809e-04, 2.615416240187202e-04, -6.418225141260302e-04, -7.266847760259519e-02, 7.580909613364026e-03, -7.260230424098423e-02, -8.052571567413480e-02, 6.521928193662712e+00, 2.891921942786487e+01, -4.302142886727015e+01, 2.364302250386845e+01, 3.575643320894192e+02, 2.572654531081801e+01, 3.464085846831062e-04, 2.417054624493788e+01, 2.871433999991601e+02, 3.213906681076925e-06, 4.404424311740082e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
