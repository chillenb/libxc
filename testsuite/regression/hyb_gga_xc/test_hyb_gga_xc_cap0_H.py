
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_cap0_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cap0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.986820010617838e-01, -4.514598778957178e-01, -2.756921877075192e-01, -1.146987297866468e-01, -1.625469820893899e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_cap0_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cap0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.580825290336053e-01, 1.127989917091438e+00, -5.846485342968413e-01, 6.042852775360686e+01, -3.359852078865555e-01, 4.055334620639832e+01, -5.468941761363298e-02, 5.525866600898257e-01, 3.638157172589236e-02, 1.896482580721351e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_cap0_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_cap0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [3.011692467250453e-04, 2.482131967249055e-02, 1.241065983624527e-02, -5.490204281018943e-03, 1.845551084427809e-02, 9.227755422139045e-03, -6.252709976144832e-02, 8.830514845194422e-02, 4.415257422597211e-02, -1.104839117290992e+01, 2.898792683101044e-01, 1.449396341550525e-01, -2.022553969497547e+05, 3.528994741687713e-03, 1.764497371443206e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
