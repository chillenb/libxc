
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_sb98_2b_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.427673820255559e+00, -1.018785722110850e+00, -3.303550938928393e-01, -1.320657418869295e-01, -6.478353686739005e-02, -1.453239026353124e-02, -3.022351186995870e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_sb98_2b_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.837817264549678e+00, -1.839276580142505e+00, -1.276774868862426e+00, -1.277705324905336e+00, -2.928714385291356e-01, -2.932694674043725e-01, -1.685530649238072e-01, 2.981123813830222e-01, -5.516777417061867e-02, 1.978534477553552e-01, -1.994414385939451e-02, -1.940276276380209e-02, -5.533553239419821e-04, 1.741397433713051e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_sb98_2b_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.071381451705591e-05, 0.000000000000000e+00, -8.042461885781581e-05, -3.453669126023314e-04, 0.000000000000000e+00, -3.441190421735182e-04, -7.091269896542653e-02, 0.000000000000000e+00, -7.071124376746510e-02, -2.879326190391252e+00, 0.000000000000000e+00, 4.787494059817580e+01, -6.643093590619836e+01, 0.000000000000000e+00, 5.731980117878154e+03, -2.303743499734848e-01, 0.000000000000000e+00, -1.404782816493179e-01, -2.778588896187313e+00, 0.000000000000000e+00, 9.787320292471630e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
