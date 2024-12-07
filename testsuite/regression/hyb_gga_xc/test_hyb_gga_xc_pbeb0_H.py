
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_pbeb0_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbeb0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.986907939368116e-01, -4.535718312551248e-01, -2.798010282594017e-01, -1.010607603309814e-01, -5.547462196707215e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_pbeb0_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbeb0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.580598621419536e-01, 1.128006376639407e+00, -5.806936347324804e-01, 6.042906610704160e+01, -3.319881096619582e-01, 4.055343968739472e+01, -1.052922001448007e-01, 5.525759030965972e-01, -7.391422146620583e-03, 1.896438080037501e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_pbeb0_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbeb0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.263400263527345e-04, 2.482160514865700e-02, 1.241080257432850e-02, -8.635196914193745e-03, 1.845559943424590e-02, 9.227799717122948e-03, -8.322112088879757e-02, 8.830508776757422e-02, 4.415254388378709e-02, -3.305758916155486e+00, 2.898735296230875e-01, 1.449367648115437e-01, -4.150520979684195e+00, 3.528911973969391e-03, 1.764455986913782e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
