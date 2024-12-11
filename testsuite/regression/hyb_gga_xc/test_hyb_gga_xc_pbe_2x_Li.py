
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_pbe_2x_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.495203926382755e-01, -6.082442080467341e-01, -1.861417928109218e-01, -8.557326134180501e-02, -3.676957812692630e-02, -9.039582149526562e-03, -1.688978270625068e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_pbe_2x_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.102761322983946e+00, -1.103574340300628e+00, -7.698124106285706e-01, -7.703203041060660e-01, -1.922629162617071e-01, -1.923463409464576e-01, -1.145554639627990e-01, -1.091314570859653e-01, -4.019844164769323e-02, 3.424535401794722e-01, -1.208123551054763e-02, -1.199442614327591e-02, -2.438284326083525e-04, -1.733398486962947e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_pbe_2x_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-6.631159733570634e-05, 9.190971700708733e-05, -6.592490834219945e-05, -2.955596895721170e-04, 2.980993506782570e-04, -2.941331096849533e-04, -2.973058959455824e-02, 6.249948659585063e-03, -2.964949926716475e-02, 1.643071807346435e+00, 6.762268918356340e+00, 3.258939195305143e+00, -1.849304749523747e+01, 2.258698854598489e+01, 1.051185509360001e+01, -1.240085571363345e-01, 3.357174600576258e-04, -1.157910138237853e-01, -5.690028989216651e-01, 3.212885779437900e-06, -8.144698014764331e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
