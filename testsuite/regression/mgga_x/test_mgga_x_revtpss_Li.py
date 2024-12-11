
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_revtpss_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.750858459191304e+00, -1.209543471002415e+00, -2.977463914237423e-01, -1.585801908648005e-01, -6.345454625449125e-02, -2.054733655625132e-02, -3.505641557981596e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_revtpss_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.349541039646137e+00, -2.351673745982466e+00, -1.628193158117273e+00, -1.629613442017412e+00, -3.936157016237703e-01, -3.932657660380642e-01, -2.124450880323105e-01, -2.609111414186119e-02, -8.274698585740911e-02, -8.296413305158503e-04, -2.750646504909760e-02, -2.723266425828722e-02, -5.541564195078383e-04, -2.260030367871350e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revtpss_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.389330387548998e-04, 0.000000000000000e+00, -3.359056735628334e-04, -9.530380481769451e-04, 0.000000000000000e+00, -9.510099071428461e-04, -1.852065716827002e-02, 0.000000000000000e+00, -1.932240529354180e-02, -7.366416789929252e+00, 0.000000000000000e+00, -4.352781648750262e-01, -2.566123635431183e+01, 0.000000000000000e+00, -2.785411081860447e+00, -1.855375773948757e-04, 0.000000000000000e+00, -4.130531457693812e-01, -1.271365602272224e-10, 0.000000000000000e+00, 9.208523377309291e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revtpss_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revtpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.323472398328504e-03, 2.304817011437339e-03, 2.364145176797173e-03, 2.368349765299592e-03, -6.355538362972272e-04, -6.726087519172267e-04, 2.683296950295916e-02, 1.181165623758641e-10, -1.556610338874278e-02, 6.061104628947540e-17, 1.320718701593170e-15, 1.354661166472653e-10, 2.484942007651403e-33, -5.495081988228646e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
