
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
    ref_tgt = [-1.881311840718756e+00, -1.313754857559319e+00, -3.398661384141171e-01, -1.361767072828414e-01, "nan", 1.723370353758604e+165, 7.314429193947192e+258]
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
    ref_tgt = [-2.488232144037742e+00, -2.488232144037742e+00, -1.728733800950400e+00, -1.728733800950400e+00, -3.871499056616308e-01, -3.871499056616308e-01, -1.805751340605709e-01, -1.805751340605709e-01, "nan", "nan", -5.744928928746077e+164, -5.744928928746081e+164, -2.438145789090372e+258, -2.438145789090368e+258]
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
    ref_tgt = [-1.221463550425863e-05, -2.442927100851726e-05, -1.221463550425863e-05, -5.272478911114859e-05, -1.054495782222972e-04, -5.272478911114859e-05, -1.595089980270093e-02, -3.190179960540187e-02, -1.595089980270093e-02, -4.647319066200530e-01, -9.294638132401060e-01, -4.647319066200530e-01, -1.883372279536649e+01, -3.766744559073297e+01, -1.883372279536649e+01, -6.645076942834539e+03, -1.329015388566908e+04, -6.645076942834539e+03, -6.655170061713856e+10, -1.331034012342771e+11, -6.655170061713856e+10]
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
    ref_tgt = [1.586654468625490e-04, 1.586654468625489e-04, 2.291986250547616e-04, 2.291986250547616e-04, 9.594972239401253e-04, 9.594972239401247e-04, 2.230713662128392e-03, 2.230713662127899e-03, 5.630050759799800e-03, 5.630050755251244e-03, 2.440459012949992e-02, 2.440459012949993e-02, 1.372955352059150e+00, 1.372955352059150e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_lp90_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_lp90", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
