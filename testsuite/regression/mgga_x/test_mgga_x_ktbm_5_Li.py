
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_5_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_5", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.862710136986197e+00, -1.251544594080361e+00, -3.011591517682365e-01, -1.707820877388656e-01, -6.183916901831149e-02, -1.378418168315496e-02, -2.535860136933960e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_5_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_5", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.593663907796278e+00, -2.596175780224034e+00, -1.786775557602877e+00, -1.788420936717570e+00, -3.721606585765821e-01, -3.711517637811665e-01, -2.344725599396834e-01, -1.609613393103387e-02, -7.691013313247387e-02, -5.105180428031511e-04, -1.692692342686026e-02, -1.680302710184064e-02, -3.409967647678578e-04, -2.449762429706863e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_5_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_5", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.586641592592527e-04, 0.000000000000000e+00, -4.570761638309225e-04, -1.786776351581759e-03, 0.000000000000000e+00, -1.781648160066580e-03, -5.269323403423386e-02, 0.000000000000000e+00, -5.475967588151066e-02, -7.085825317599637e+00, 0.000000000000000e+00, -3.701954790588323e+01, -7.146469947291926e+01, 0.000000000000000e+00, -9.285234836311218e+04, -6.893711170752033e-01, 0.000000000000000e+00, -3.309694942748272e+01, -1.406054936538045e+00, 0.000000000000000e+00, -1.286826551398837e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_5_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_5", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.586287567846464e-02, 1.585401168716002e-02, 1.737904689239746e-02, 1.739113930837635e-02, -5.406612928753335e-03, -5.592254627437832e-03, 1.904172176562293e-01, 4.729183999966926e-04, -4.640388236305234e-02, 3.783135865655302e-05, 1.023802721055920e-05, 4.810066528423081e-04, 1.707176425796083e-10, -1.196111954803603e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
