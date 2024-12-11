
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_js18_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_js18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.592508008636940e+00, -1.105826704359424e+00, -3.338233518394548e-01, -1.540821267243600e-01, -6.600804253548918e-02, -6.539827864777392e-02, -2.689435505691506e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_js18_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_js18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.130565566672905e+00, -2.132468405266604e+00, -1.481234507513534e+00, -1.482509202014733e+00, -3.312697613550782e-01, -3.300091725747386e-01, -2.045968390227294e-01, -2.085548586195447e-02, -7.298292108270626e-02, -2.445477041342156e-03, -5.252197256582470e-02, -2.155063733374301e-02, -2.074010790932335e-02, -3.277364439094707e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_js18_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_js18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.785332920939453e-04, 0.000000000000000e+00, -3.772301590522335e-04, -9.491995350046553e-04, 0.000000000000000e+00, -9.476691357618636e-04, -4.784928727854627e-03, 0.000000000000000e+00, -5.349357225290793e-03, -9.194215240602267e+00, 0.000000000000000e+00, -5.044404203727495e+02, -2.187360277387177e+01, 0.000000000000000e+00, -3.497865617963190e+06, -1.653208398732213e+01, 0.000000000000000e+00, -4.453489931123190e+02, -6.730896722835238e+02, 0.000000000000000e+00, 1.408189298649457e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_js18_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_js18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.017590352347768e-03, 1.994407236090869e-03, 1.895904845932920e-03, 1.899443174097638e-03, -2.011876764683309e-02, -2.043715073254769e-02, 2.636341402309236e-02, 3.893714147103896e-03, -1.236805967245171e-01, 7.367690055462676e-04, 1.275660390956486e-04, 3.911474973222833e-03, 4.224649172011609e-08, -5.528679824704181e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
