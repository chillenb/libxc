
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_revscan0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revscan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.528059007466110e+00, -1.059806424749198e+00, -2.447289389325787e-01, -1.380431247608255e-01, -5.388042162188120e-02, -4.422245030613760e-03, -1.620948927786289e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_revscan0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revscan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.007624887313044e+00, -2.009484528005558e+00, -1.377353889485477e+00, -1.378603091296087e+00, -2.071835456968192e-01, -2.428137445452284e-01, -1.825839291982597e-01, 1.287661275094536e+00, -6.288940620444530e-02, 4.981228704018329e+00, -7.032142388860414e-03, 1.278942556090279e+00, -3.465933790750988e-05, 6.088514626392834e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_revscan0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revscan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.410486523754864e-04, 0.000000000000000e+00, -1.405379748659025e-04, -5.512098747785739e-04, 0.000000000000000e+00, -5.487659238395152e-04, -1.747781040912267e-01, 0.000000000000000e+00, -1.260433459820657e-01, -2.257003079322410e+00, 0.000000000000000e+00, -3.317481859583368e+04, -6.124672549396816e+01, 0.000000000000000e+00, -1.010222637113118e+10, 1.509106393662558e+01, 0.000000000000000e+00, -2.822736288751075e+04, 2.416323468972164e+04, 0.000000000000000e+00, -1.177232519499346e+11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_revscan0_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revscan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_revscan0_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_revscan0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [7.601063564252945e-03, 7.594012568928375e-03, 1.016707259026509e-02, 1.014985827252883e-02, 4.422337937795694e-02, 3.252201568407223e-02, 8.835695400558818e-02, 4.239675004846063e-01, 1.594042326562183e-01, 4.116012899865773e+00, 1.458432331984100e-11, 4.104035726836691e-01, 1.756490409980167e-23, 5.135466988975399e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
