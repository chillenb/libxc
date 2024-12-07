
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_wb97m_v_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_wb97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.597712545616571e+00, -1.041104860622169e+00, -2.115583605629386e-01, 3.024363187280334e-02, -1.643346036471813e-02, -2.048828171531131e-02, -5.734761532274114e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_wb97m_v_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_wb97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.222618475472248e+00, -2.224206407107547e+00, -1.401481347482882e+00, -1.402537593090162e+00, -2.148721723698598e-01, -2.146070236846275e-01, 6.705273898598219e-02, -5.661734367534479e-02, -3.950665845194983e-03, 1.380171708793794e-01, -2.575476284018035e-02, -2.568598034427914e-02, -7.529253631432531e-04, -7.664242003884000e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_wb97m_v_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_wb97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.825719108745869e-04, 0.000000000000000e+00, -1.822602667068102e-04, -6.855420571359739e-04, 0.000000000000000e+00, -6.836024990353993e-04, -2.362924958492028e-02, 0.000000000000000e+00, -2.335417357000266e-02, -2.033245280208699e+00, 0.000000000000000e+00, 1.236236706099287e+01, 3.022355118122764e+00, 0.000000000000000e+00, 2.281626973135194e+03, -4.112996086441929e-02, 0.000000000000000e+00, -4.124602395470885e-02, -1.704284343310842e-01, 0.000000000000000e+00, -7.894808148347306e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_wb97m_v_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_wb97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_wb97m_v_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_wb97m_v", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.956442689192411e-02, 1.957552730434033e-02, 1.350730748920062e-02, 1.351433728934576e-02, -9.571044711568348e-03, -9.845626904729262e-03, -1.602163429886086e+00, -8.198162299836826e-06, -1.732474509141188e-01, -2.354794115980302e-08, -1.001256070554945e-08, -2.230998898961713e-05, -5.377223905916151e-20, -9.987564565192595e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
