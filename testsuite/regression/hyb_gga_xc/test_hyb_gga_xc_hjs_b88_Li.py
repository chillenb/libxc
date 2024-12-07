
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hjs_b88_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.410285626035856e+00, -1.005868802270256e+00, -3.151069907188542e-01, -1.481877384401328e-01, -7.057739265299037e-02, -1.484051586680628e-01, -1.790559887859811e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hjs_b88_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.829442127905315e+00, -1.830901166989108e+00, -1.277596005773601e+00, -1.278526320780786e+00, -2.858964424333486e-01, -2.857825241547964e-01, -1.927924631672419e-01, 6.841036832920423e-02, -7.349399282644008e-02, 3.428185832405125e-01, 8.080830011141683e-02, 1.182754767492511e-01, -1.135871182268017e-15, -1.343631423570552e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hjs_b88_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hjs_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.298797756956680e-05, 9.190971700708733e-05, -9.252087194268045e-05, -3.854314762513023e-04, 2.980993506782570e-04, -3.837347075085908e-04, -6.526307008685595e-02, 6.249948659585063e-03, -6.525375794741371e-02, 9.491175938087699e-01, 6.762268918356340e+00, -3.497578199773733e+03, -3.876340874036356e+01, 2.258698854598489e+01, 1.129349427299244e+01, -2.258614392123717e+03, 3.357174600576258e-04, -2.586688185296905e+03, 1.606543586949356e-06, 3.212885779437900e-06, 1.606543586949356e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
