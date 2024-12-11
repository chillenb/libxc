
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_hle17_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_hle17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.235305034307935e+00, -1.553786750063031e+00, -3.969820201428171e-01, -2.072683039844693e-01, -8.479773962023285e-02, -2.568848104904669e-02, -4.382051947486224e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_hle17_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_hle17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.988261473412594e+00, -2.990846847311434e+00, -2.081514144538725e+00, -2.083220145538384e+00, -5.203423150333766e-01, -5.199266609109491e-01, -2.760645160190235e-01, -9.480005829011048e-02, -1.099861021408750e-01, -3.680076468659035e-02, -3.438382126989008e-02, -3.407487024716611e-02, -6.926955243898063e-04, -2.825037959839114e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_hle17_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_hle17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.600328057346187e-04, 1.386252986746217e-04, -3.567019741052797e-04, -9.037477560183132e-04, 5.969146803740697e-04, -9.012383772281139e-04, 4.011080286037011e-02, 1.796789276683559e-01, 3.922198058731342e-02, -7.238528984601961e+00, 4.170868700560120e+00, 1.737871345611303e+00, 4.337898265053650e+01, 1.681681181378362e+02, 8.186343636198251e+01, -1.479189851830504e-04, 5.396702373945528e-09, -3.298489699367110e-01, -1.013563174152062e-10, -9.615016850107889e-16, -1.946747329801935e+11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_hle17_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_hle17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.904597001446921e-03, 2.881020674112530e-03, 2.955181470996466e-03, 2.960437206624490e-03, -7.944422953715340e-04, -8.407609398965334e-04, 3.354121168884015e-02, -1.059214120129484e-10, -1.945762923592848e-02, 4.346345323501434e-17, 9.468964273257145e-16, 9.621308438796742e-11, 1.781963402376420e-33, -6.868852485291018e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
