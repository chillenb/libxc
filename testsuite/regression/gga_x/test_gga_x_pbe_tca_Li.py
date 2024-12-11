
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbe_tca_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_tca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.795240009591995e+00, -1.285839297436372e+00, -4.453211014216507e-01, -1.600529387113422e-01, -8.313951086234667e-02, -2.534820408000604e-02, -4.738652697643957e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbe_tca_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_tca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.239364371708001e+00, -2.241506328105633e+00, -1.510834141737964e+00, -1.512213207276394e+00, -3.604578474055060e-01, -3.606426062080414e-01, -2.052057861119818e-01, -3.219194136900368e-02, -7.035211971523964e-02, -1.024173368250214e-03, -3.383986277503713e-02, -3.359932882995867e-02, -6.840923521170950e-04, -4.863277443257219e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbe_tca_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_tca", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.626600397022972e-04, 0.000000000000000e+00, -2.617419882994840e-04, -1.069193241728132e-03, 0.000000000000000e+00, -1.065696830830373e-03, -1.131199841559596e-01, 0.000000000000000e+00, -1.129318628026656e-01, -4.018387503435036e+00, 0.000000000000000e+00, -6.459687523863192e-01, -8.862469879854324e+01, 0.000000000000000e+00, -4.137416367052889e+00, -6.563435295309789e-01, 0.000000000000000e+00, -6.129516169130299e-01, -3.011892778747896e+00, 0.000000000000000e+00, -4.311216793105818e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
