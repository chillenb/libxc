
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_lp90_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_lp90", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.140904148019771e+00, -1.319261517814559e+00, -3.243708564419219e-01, -1.377572658799071e-01, -5.566814624293979e-02, 7.075113846959833e+00, 7.237265901692925e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_lp90_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_lp90", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.400880485804096e+00, -2.400880485804096e+00, -1.726886122743840e+00, -1.726886122743840e+00, -3.923232320181764e-01, -3.923232320181764e-01, -1.800479193483484e-01, -1.800479193483377e-01, -6.797291327173383e-02, -6.797291325164688e-02, -1.777077874277563e+00, -1.777077874277567e+00, -2.412275852671808e+05, -2.412275852671808e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_lp90_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_lp90", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.221463550425863e-05, -2.442927100851726e-05, -1.221463550425863e-05, -5.272478911114859e-05, -1.054495782222972e-04, -5.272478911114859e-05, -1.595089980270093e-02, -3.190179960540187e-02, -1.595089980270093e-02, -4.647319066200530e-01, -9.294638132401060e-01, -4.647319066200530e-01, -1.883372279536649e+01, -3.766744559073297e+01, -1.883372279536649e+01, -6.645076942834539e+03, -1.329015388566908e+04, -6.645076942834539e+03, -6.655170061713856e+10, -1.331034012342771e+11, -6.655170061713856e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_lp90_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_lp90", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([1.586654468625490e-04, 1.586654468625489e-04, 2.291986250547616e-04, 2.291986250547616e-04, 9.594972239401253e-04, 9.594972239401247e-04, 2.230713662128392e-03, 2.230713662127899e-03, 5.630050759799800e-03, 5.630050755251244e-03, 2.440459012949992e-02, 2.440459012949993e-02, 1.372955352059150e+00, 1.372955352059150e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
