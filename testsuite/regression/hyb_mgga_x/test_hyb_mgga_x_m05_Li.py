
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m05_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.217188005140602e+00, -8.120832934894909e-01, -2.539421648933490e-01, -1.193122740773571e-01, -4.903845005413603e-02, -1.330993032888760e-03, -2.049777557841214e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m05_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.286760709707245e+00, -2.290581179819272e+00, -7.788885181103042e-01, -7.795143940808633e-01, -1.820738976061503e-01, -1.842019220867825e-01, -1.390521410585177e-01, -2.196876942594075e-03, -6.471624916310351e-02, -4.877316645240695e-05, -1.631322014332269e-03, -2.335561104924698e-03, -3.254183836273748e-05, -1.282713969612214e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m05_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.947124636064147e-04, 0.000000000000000e+00, -1.941844121773641e-04, -8.059097703101548e-04, 0.000000000000000e+00, -8.027226583133592e-04, -3.164884137247013e-01, 0.000000000000000e+00, -3.142040921094830e-01, -3.158178281765474e+00, 0.000000000000000e+00, -1.944863430452055e-02, -1.325399689550626e+02, 0.000000000000000e+00, -1.043702328456849e-01, -6.979422624464286e-06, 0.000000000000000e+00, -1.863902279483957e-02, -4.761502962110945e-12, 0.000000000000000e+00, -2.056777779582271e+11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m05_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m05", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [9.460482433169903e-02, 9.480245713590041e-02, -4.440281702416578e-02, -4.439260799086645e-02, -2.872171111634994e-02, -2.846710853846081e-02, -5.713556244146809e-01, 7.222183737150911e-05, -5.596174540934534e-03, 1.483347150048220e-08, 3.601088443522743e-08, 7.793732444740785e-05, 2.021692592647896e-19, 9.173603848963275e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
