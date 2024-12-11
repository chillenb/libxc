
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_o3lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_o3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.677305350577870e+00, -1.189119399942117e+00, -4.028865194897232e-01, -1.489788176291167e-01, -7.443208813413955e-02, -2.775881775490824e-02, -5.186251125932732e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_o3lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_o3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.190051323190740e+00, -2.191874084847382e+00, -1.500723991402009e+00, -1.501889218540116e+00, -3.538733933704785e-01, -3.544864193668533e-01, -1.970187350452358e-01, -1.225790808654419e-01, -5.458250344268985e-02, -3.967700513824386e-02, -3.688588447534963e-02, -3.674258247309307e-02, -7.221111617559083e-04, -6.051669324234116e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_o3lyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_o3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.139285056017273e-05, 4.230480491699886e-06, -5.120329626043158e-05, -3.853678520381011e-04, 2.954022849291356e-05, -3.841423419241732e-04, -1.083466522020522e-01, 3.866747919504811e-02, -1.080092702569884e-01, -5.295330557193809e-01, 3.722869163256963e+00, 2.146215544243291e+00, -9.709438723461740e+01, 1.909121184807025e+01, 1.017782753028300e+01, -6.244721928715306e-01, 6.428238830639903e-02, -5.807961569413350e-01, -3.014193174119767e+00, 0.000000000000000e+00, -4.314507930151614e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
