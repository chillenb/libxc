
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_sb98_1b_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_1b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.472945836902575e+00, -1.047686336688387e+00, -3.543538207535775e-01, -1.395549300773573e-01, -7.113018932445395e-02, -1.989169289673585e-02, -4.320063631582262e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_sb98_1b_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_1b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.923634002111092e+00, -1.925205134473625e+00, -1.332301543467265e+00, -1.333277991209741e+00, -2.686891906640453e-01, -2.690397858845306e-01, -1.852633809927155e-01, 2.996023278588691e-01, -5.716136060525871e-02, 2.026886895265719e-01, -2.689736140513682e-02, -2.630340928360719e-02, -7.420858095676511e-04, -1.074727862843100e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_sb98_1b_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_1b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.304083168452647e-05, 0.000000000000000e+00, -3.280682652669180e-05, -2.400074180539985e-04, 0.000000000000000e+00, -2.390138003525004e-04, -9.677907585947959e-02, 0.000000000000000e+00, -9.660348113925808e-02, 6.991279352662840e-01, 0.000000000000000e+00, 4.893928150804019e+01, -7.866999926104981e+01, 0.000000000000000e+00, 5.891707914861900e+03, -5.260501572486251e-01, 0.000000000000000e+00, -4.145241063121036e-01, -4.202937706126607e+00, 0.000000000000000e+00, 8.135495207689315e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
