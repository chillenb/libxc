
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_tpss_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.935022555051660e-02, -4.020265710256662e-02, -4.001442952053117e-03, -2.468466642692871e-03, -6.150425750701574e-09, -7.610696121655294e-09, -1.790424362528482e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_tpss_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.055077402976554e-01, -1.054751965646466e-01, -8.828641931393161e-02, -8.826116296955550e-02, -1.836433780316002e-02, -1.845970773765164e-02, -2.922102814574210e-02, -3.961342806760824e-01, -2.696301644891019e-03, -9.668284287397352e-03, -4.884014359418227e-08, -4.993402974784920e-08, -1.136104462411605e-15, -1.343836472708371e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpss_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([8.317826087057291e-05, 1.933562908123026e-04, 8.321088485355910e-05, 2.081323787754584e-04, 3.188370920880546e-04, 2.080386470954447e-04, 1.724694231516702e-02, -2.337108015954634e-02, 1.729706043281402e-02, 3.293822939641726e+01, 8.871800675491191e+01, 4.555906597740740e+02, 1.573355919234538e+01, 3.144581314832332e+01, 6.254215312780487e+04, 4.337802437314366e-04, -2.079140542073983e-04, 4.456990636845129e-04, 1.606947891974255e-06, 3.213895784120031e-06, 1.606947891982351e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpss_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-7.229015793851551e-03, -7.229015793851550e-03, -3.378212368436674e-03, -3.378212368436672e-03, 5.395029167043902e-04, 5.395029167043899e-04, -1.137154461695792e+00, -1.137154461695540e+00, -3.762624502470487e-02, -3.762624499430638e-02, 2.604546488369221e-14, 2.604546488369220e-14, -9.070442358332176e-32, -9.070442358332180e-32])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
