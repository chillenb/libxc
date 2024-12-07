
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbetrans_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbetrans", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.795105210953902e+00, -1.285660751695045e+00, -4.450323821296815e-01, -1.600441864682590e-01, -8.310372455678466e-02, -2.065809966577179e-02, -3.859865135599423e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbetrans_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbetrans", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.239524230502044e+00, -2.241666090922431e+00, -1.511020437441467e+00, -1.512399524136765e+00, -3.618005334723197e-01, -3.620314541681260e-01, -2.052148406528835e-01, -2.625955220981753e-02, -7.034953360142977e-02, -8.342421535811641e-04, -2.760833163630019e-02, -2.741000397315969e-02, -5.572273247141549e-04, -3.961379747991888e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbetrans_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbetrans", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.620779906568930e-04, 0.000000000000000e+00, -2.611619133803057e-04, -1.066957471880442e-03, 0.000000000000000e+00, -1.063468071219386e-03, -1.122827398484281e-01, 0.000000000000000e+00, -1.120721467301513e-01, -4.009268064088181e+00, 0.000000000000000e+00, -2.853267315589949e-01, -8.852594617945763e+01, 0.000000000000000e+00, -1.825199531123691e+00, -2.899515102388465e-01, 0.000000000000000e+00, -2.707641037845864e-01, -1.328678045033550e+00, 0.000000000000000e+00, -1.901865948327759e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
