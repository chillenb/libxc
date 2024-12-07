
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lc_wpbesol_whs_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbesol_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.516605633883708e+00, -9.880220111765085e-01, -1.244472220484205e-01, -3.402388377909610e-02, -3.509401759462614e-03, -8.041042295514813e-06, -5.500172210097695e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lc_wpbesol_whs_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbesol_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.072504349100424e+00, -2.074463586295203e+00, -1.352026135424395e+00, -1.353266323397315e+00, -1.409670274761173e-01, -1.409313050945265e-01, -5.753959198813242e-02, -1.061367979834917e-01, -1.217748313376318e-02, 4.670283516620314e-01, -1.634080194606423e-05, -1.598874023076609e-05, -1.324511790031402e-10, -4.751615834263712e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lc_wpbesol_whs_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbesol_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.929363540813710e-05, 8.057094003003931e-05, -8.885216588704056e-05, -3.682547619395984e-04, 2.747845524533765e-04, -3.666770942409949e-04, -4.122705133171500e-02, 9.764292986584010e-03, -4.119160958894086e-02, 2.379384422484335e+00, 4.978936063600431e+00, 2.489466449592250e+00, 1.580807492335498e+01, 3.188247312456189e+01, 1.594123656227894e+01, 3.507909179300459e-04, 7.058310792173810e-04, 3.510097169473599e-04, 3.380088207956301e-06, 6.759926358451161e-06, 3.380088393887239e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
