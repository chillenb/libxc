
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m06_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.938727145795362e-02, -3.979161498179351e-02, 2.311279694031998e-02, 7.454321734617630e-04, 1.105142429460965e-07, 4.412790887992715e-02, 1.258657752993939e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m06_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.322861546608589e-02, -1.311913009316190e-02, -2.419638220839304e-02, -2.424772090917005e-02, -1.156954693825214e-01, -1.134346401028973e-01, 7.658485537172056e-03, 1.113369661165436e+00, -1.038554713612288e-02, 6.607666334539675e-01, 7.251609369639631e-02, 6.611617815487840e-02, 1.672351392168945e-03, 2.346414414012601e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.003309127082516e-04, 0.000000000000000e+00, -1.001722175735531e-04, -7.493422607372057e-05, 0.000000000000000e+00, -7.412430505689947e-05, 1.274816190080865e-02, 0.000000000000000e+00, 1.484959305333587e-02, -9.267195615235719e+00, 0.000000000000000e+00, -5.587754590117738e+02, 6.060274612081903e+01, 0.000000000000000e+00, -2.873454447708166e+06, -2.459744360598899e+00, 0.000000000000000e+00, -6.206027331312889e+02, -2.650269553108446e+01, 0.000000000000000e+00, -1.360065462588579e+07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([4.061839897528322e-03, 4.076615597865625e-03, -8.096168816562296e-04, -7.968849118389440e-04, 2.114175647002824e-02, 2.022449624064341e-02, 3.385415938928748e-01, 9.463775835436167e-03, -1.449299210565063e-01, 1.179505559453673e-03, 3.363435281533769e-06, 9.176813190464379e-03, 4.305169104495417e-14, 5.933150841070762e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
