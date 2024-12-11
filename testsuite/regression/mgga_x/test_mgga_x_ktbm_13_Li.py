
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_13_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_13", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.979202132559271e+00, -1.304005287679139e+00, -2.466200484044705e-01, -1.824811615229101e-01, -5.350503695052086e-02, -1.036017093913373e-02, -1.923265959818500e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_13_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_13", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.808721060672174e+00, -2.811424232850610e+00, -1.957197298919227e+00, -1.958899935975813e+00, -3.241221986119099e-01, -3.242203775273165e-01, -2.529201286704529e-01, -1.242809782451246e-02, -7.564236385421820e-02, -3.941027696732207e-04, -1.306706766110131e-02, -1.297405858553602e-02, -2.632379102511187e-04, -1.929027898490818e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_13_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_13", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.943335156487454e-04, 0.000000000000000e+00, -6.919833220981244e-04, -2.569772193476272e-03, 0.000000000000000e+00, -2.563069960363610e-03, -3.769180227725034e-02, 0.000000000000000e+00, -3.959245281776207e-02, -1.091883715881966e+01, 0.000000000000000e+00, -1.942568928889160e+01, -6.738203262534151e+01, 0.000000000000000e+00, -4.865293977734023e+04, -3.612285597107964e-01, 0.000000000000000e+00, -1.736889204403527e+01, -7.367444597062058e-01, 0.000000000000000e+00, 6.588909827450229e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_13_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_13", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.439539830996918e-02, 2.436868961128781e-02, 3.208650860064795e-02, 3.208059289601591e-02, -8.755892084880808e-04, -8.366832011564174e-04, 2.712751398076206e-01, 2.485251823518829e-04, 3.602304482692091e-02, 1.982301739114614e-05, 5.364878448203582e-06, 2.528211264035374e-04, 8.945260535533034e-11, -6.638351275253411e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
