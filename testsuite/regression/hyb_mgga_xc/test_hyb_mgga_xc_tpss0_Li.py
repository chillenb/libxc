
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_tpss0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.526032960407857e+00, -1.065584015552439e+00, -3.039871254721769e-01, -1.351026866172665e-01, -5.748751204793627e-02, -1.540836721630357e-02, -2.878940233985846e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_tpss0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.910829275391507e+00, -1.912426042653784e+00, -1.368076726019213e+00, -1.369361532059979e+00, -2.815317838625760e-01, -2.821438178050954e-01, -1.895617175198106e-01, -4.157210497350823e-01, -5.860202210067898e-02, -1.029051677774083e-02, -2.059293793154635e-02, -2.044497207220321e-02, -4.156166464749472e-04, -2.954656511792497e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpss0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.594905593824951e-04, 1.933562908123026e-04, -6.588149064094997e-04, -9.784186373323887e-04, 3.188370920880546e-04, -9.764305035419383e-04, -5.013502558839079e-02, -2.337108015954634e-02, -4.959478223238094e-02, 1.189644488357855e+01, 8.871800675491191e+01, 4.553821217580089e+02, -3.216189351663604e+01, 3.144581314832332e+01, 6.254082075410650e+04, -2.115095753516990e-01, -2.079140542073983e-04, -1.974636845173528e-01, -9.699154997518689e-01, 3.213895784120031e-06, -1.388334334959962e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpss0_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_tpss0_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_tpss0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.127040913358025e-02, 3.136659900080805e-02, 1.686831252647410e-02, 1.692841068214170e-02, 9.331907347370800e-04, 8.701700967764567e-04, -3.779296660026987e-01, -1.137154461645177e+00, -2.805233081896365e-02, -3.762624499430635e-02, 2.608173215244415e-14, 5.775389609766414e-11, -9.070436415254624e-32, 4.530824741917106e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
