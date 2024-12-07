
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_m06_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.109143554202172e+00, -1.413294972747361e+00, -3.350928079374445e-01, -1.862786958980162e-01, -7.092815182892111e-02, -9.111902697136208e-02, -1.707706780539315e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_m06_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.156560351461420e+00, -3.157810676297607e+00, -1.667696997331326e+00, -1.668639575933949e+00, -3.768689488152914e-01, -3.904952504472123e-01, -1.653542647911569e-01, -1.147776647832923e-01, -7.795262675531676e-02, -3.690797565161812e-03, -1.220857989706274e-01, -1.197162972091348e-01, -2.465323291676869e-03, -1.752608283484649e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_m06_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.395997377065974e-04, 0.000000000000000e+00, -3.385207899708005e-04, -1.279759140694216e-03, 0.000000000000000e+00, -1.275476986844759e-03, -3.854968972608994e-02, 0.000000000000000e+00, -3.624108539555278e-02, -5.199581131883313e+00, 0.000000000000000e+00, 3.689483856493438e-01, -6.340076297316975e+01, 0.000000000000000e+00, 2.382516687017495e+00, -1.346987378740997e+00, 0.000000000000000e+00, 3.499139806458657e-01, -5.753143997126903e+00, 0.000000000000000e+00, 2.482638524811092e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_m06_l_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_m06_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_m06_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [7.738724413864913e-02, 7.711319791641573e-02, 3.943877135519920e-03, 3.853792797360242e-03, 1.756457585378796e-03, 3.433517680413688e-03, -2.059042263753248e+00, -1.803586846019249e-04, 1.034500550408005e-01, -3.696306052476634e-08, -1.108217960548471e-07, -1.946581160160833e-04, -9.684567296202895e-19, -4.123819446089314e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
