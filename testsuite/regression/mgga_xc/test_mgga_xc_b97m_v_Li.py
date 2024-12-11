
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_b97m_v_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_b97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.169141410863980e+00, -1.403537929868118e+00, -3.010680358289813e-01, -1.481659973713850e-01, -9.336672010384876e-02, -2.614608165346933e-02, -7.383010409321017e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_b97m_v_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_b97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.210404362187689e+00, -3.213248187848712e+00, -2.146908328623067e+00, -2.148758700529018e+00, -3.991611962679236e-01, -3.994805458729609e-01, -1.922786786091420e-01, 3.469626431359809e-01, -9.740093670993660e-02, 2.057428644123299e-01, -3.345883050897103e-02, -3.322040389972723e-02, -9.990059762446596e-04, -9.082548905800112e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_b97m_v_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_b97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.521355602664322e-04, 0.000000000000000e+00, -9.671900479142468e-04, -3.067962370086015e-03, 0.000000000000000e+00, -3.061754213187977e-03, 3.298872464091109e-01, 0.000000000000000e+00, 3.240619415775241e-01, -1.618387324593315e+01, 0.000000000000000e+00, 1.096069432472150e+02, 7.496992473222389e+01, 0.000000000000000e+00, 8.132862207499004e+03, 2.003849021338776e-04, 0.000000000000000e+00, -7.408714717136650e-01, 6.327259855975302e-12, 0.000000000000000e+00, 6.833199589125181e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_b97m_v_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_b97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [4.604615387326860e-02, 4.604135705710898e-02, 4.200233873205714e-02, 4.205269200401648e-02, 3.117831866902364e-03, 3.285788968393261e-03, -4.241489339564265e-01, 1.027434953073133e-05, -1.743254055852345e-01, -1.921132224304767e-07, 2.263486395366322e-09, 4.700748287797341e-06, 3.179583855761165e-20, -4.566712532777354e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
