
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m06_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.293980393097424e+00, -9.123649316191025e-01, -3.232955527511611e-01, -1.131702108622874e-01, -6.218574871332468e-02, -5.236726892455125e-02, -9.804935449691651e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m06_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.070439509617931e+00, -2.071765084034919e+00, -8.123251760004938e-01, -8.123560605671339e-01, -1.242035570035333e-01, -1.356349893580554e-01, -7.120639680769467e-02, -6.611633497839299e-02, -5.552318235323528e-02, -2.119118005169236e-03, -7.005340039241015e-02, -6.897516458302320e-02, -1.415484786126496e-03, -1.006275764624512e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.247364315182349e-04, 0.000000000000000e+00, -1.243433129442413e-04, -5.681010252096029e-04, 0.000000000000000e+00, -5.655381723079722e-04, -1.540135790321984e-01, 0.000000000000000e+00, -1.512862809302809e-01, -1.744610643723069e+00, 0.000000000000000e+00, -2.322943753020586e+00, -9.102467072457158e+01, 0.000000000000000e+00, -1.491954004368821e+01, -7.984715128135209e-01, 0.000000000000000e+00, -2.203836312383504e+00, -3.303212955568887e+00, 0.000000000000000e+00, -1.554635051465136e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m06_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([5.958827975944149e-02, 5.948079316887546e-02, -4.329268653844339e-02, -4.340085247597326e-02, 1.998275330766642e-03, 3.102144320442582e-03, -2.165849582648881e+00, -3.692363413690218e-05, 1.189479001968225e-01, -7.551211492311454e-09, -1.503912182159637e-07, -3.985627257994662e-05, -1.124578642856215e-18, -8.424546634562694e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
