
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_pbe50_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe50", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.571837870324511e-01, -6.852601784310904e-01, -2.111049480083279e-01, -9.517473573226576e-02, -4.159977388123695e-02, -1.027225140495748e-02, -1.919293489346424e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_pbe50_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe50", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.237318007978300e+00, -1.238259049668808e+00, -8.609518783370799e-01, -8.615419150105379e-01, -2.163088267491177e-01, -2.164028559887378e-01, -1.268740784407352e-01, -1.106984012019947e-01, -4.478250219508307e-02, 3.424037615802394e-01, -1.372867000071012e-02, -1.363002295871051e-02, -2.770777643275184e-04, -1.969771007910607e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_pbe50_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe50", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.162065949560409e-05, 9.190971700708733e-05, -8.118124018480078e-05, -3.561882393327415e-04, 2.980993506782570e-04, -3.545671258246009e-04, -3.421089376606062e-02, 6.249948659585063e-03, -3.411874566674984e-02, 1.406063263914834e+00, 6.762268918356340e+00, 3.242276204777003e+00, -2.255484864545065e+01, 2.258698854598489e+01, 1.040526793277377e+01, -1.409417047544747e-01, 3.357174600576258e-04, -1.316035873538505e-01, -6.465944223941995e-01, 3.212885779437900e-06, -9.255340843882541e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
