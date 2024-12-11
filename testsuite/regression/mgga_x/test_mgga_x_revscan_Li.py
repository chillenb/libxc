
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_revscan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.980140064083595e+00, -1.310740036674793e+00, -2.275478659459323e-01, -1.812456401298904e-01, -5.100901873529352e-02, -4.816589062984348e-03, -3.421776249785846e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_revscan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.757515090632454e+00, -2.760063067692244e+00, -2.010179524466313e+00, -2.011210105084609e+00, -3.161866749387497e-01, -3.166824844849362e-01, -2.472952762351831e-01, 1.716881700126048e+00, -7.579437829196066e-02, 6.172730388055123e+00, 3.742739776086458e+01, 1.705256741453706e+00, 6.122069233707052e+04, -1.663213460529610e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revscan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.572018118828362e-04, 0.000000000000000e+00, -3.558193791912388e-04, -2.573714698411140e-03, 0.000000000000000e+00, -2.559021956139352e-03, -4.599916476029940e-01, 0.000000000000000e+00, -4.605409320024599e-01, -4.382140816296786e+00, 0.000000000000000e+00, -4.423309146111157e+04, -2.127285520431616e+02, 0.000000000000000e+00, -1.251867585876348e+10, -1.699834299576496e+04, 0.000000000000000e+00, -3.763648385001434e+04, -2.814777798891071e+09, 0.000000000000000e+00, -4.667966258281290e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_revscan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_revscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.689987697137463e-02, 1.688063923600132e-02, 3.852622846243670e-02, 3.841322682529387e-02, 2.334767124296933e-03, 2.463454166749712e-03, 1.580874600213445e-01, 5.652900006461418e-01, 6.516130490718909e-02, 5.100562734975588e+00, 2.524498868257773e-01, 5.472047635782255e-01, 3.417592141866564e-01, 2.745442242238646e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
