
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_mn15_l_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.979359276939758e-02, -8.807261581537042e-02, -4.005912614072262e-01, -1.566560528088560e-02, -6.903043906775334e-02, -1.796635978177092e-02, -4.458778983378974e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_mn15_l_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [3.089725894448663e-02, 3.103540144794703e-02, -1.365704900494018e-01, -1.364245618363953e-01, 9.405247275718888e-02, 9.370088591706625e-02, 4.488549021026213e-03, -8.498228067877221e-02, 2.518285085850831e-02, -3.428161958437158e-01, -2.257649213106429e-02, -2.282990417640682e-02, -5.245294618555623e-04, -7.696734983240676e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn15_l_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.551835788938456e-04, -5.103671577876913e-04, -2.551835788938456e-04, -8.755027852745797e-04, -1.751005570549159e-03, -8.755027852745797e-04, 1.817838484416749e+00, 3.635676968833498e+00, 1.817838484416749e+00, -7.876230295256418e+00, -1.575246059051284e+01, -7.876230295256418e+00, 1.422737152799936e+03, 2.845474305599872e+03, 1.422737152799936e+03, 2.241765643810350e-07, 4.483531408987895e-07, 2.241765643810350e-07, 2.361372743918628e-15, -4.177033039197283e-14, 2.361372743918628e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_mn15_l_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_mn15_l", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.684226482225370e-02, -1.684226482225370e-02, 5.734081321769895e-03, 5.734081321769895e-03, -1.014623123003524e-01, -1.014623123003523e-01, -6.160724067560679e-01, -6.160724067559339e-01, -9.020040039684328e-01, -9.020040033466726e-01, -7.613790153269783e-08, -7.613790153269784e-08, -2.011643110608801e-19, -2.011933913204378e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
