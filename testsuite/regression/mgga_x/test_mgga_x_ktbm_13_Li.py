
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
    ref_tgt = numpy.asarray([-2.136890845874251e+00, -1.499894017430566e+00, -2.868868506641454e-01, -1.910903953182589e-01, -6.721965986966019e-02, -1.020885674953684e-02, -1.890748295993249e-04])
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
    ref_tgt = numpy.asarray([-2.619325927765223e+00, -2.621839037194784e+00, -1.769286264436752e+00, -1.770598056112964e+00, -3.709518485016626e-01, -3.729289522991978e-01, -2.420217167672315e-01, -1.242809782451247e-02, -8.852629022144554e-02, -3.941027702393399e-04, -1.346770232905349e-02, -1.297405858553602e-02, -2.713468293991789e-04, -1.871381225392066e-04])
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
    ref_tgt = numpy.asarray([-7.494199299608659e-04, 0.000000000000000e+00, -7.468732130262404e-04, -2.946231289981252e-03, 0.000000000000000e+00, -2.938347247970961e-03, -6.442292653745496e-02, 0.000000000000000e+00, -6.783481175734560e-02, -1.143653331340679e+01, 0.000000000000000e+00, -1.942568928889160e+01, -9.418311226768674e+01, 0.000000000000000e+00, -4.865293819454445e+04, 9.728775685251032e-02, 0.000000000000000e+00, -1.736889204403527e+01, 2.203852400616015e-01, 0.000000000000000e+00, -2.202594202417216e+05])
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
    ref_tgt = numpy.asarray([2.968385469751945e-02, 2.964835939536362e-02, 4.819621565615440e-02, 4.816210257095715e-02, 2.225847398899616e-02, 2.407466124243129e-02, 3.032128504205986e-01, 2.485251823518829e-04, 3.511688448310827e-01, 1.982301621494738e-05, -2.968152927730106e-08, 2.528211264035374e-04, -2.118826602788925e-16, 9.608427487516284e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
