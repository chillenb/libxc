
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_perdew_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_perdew", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.633558506039394e+01, 8.104182978448975e+00, 7.988262005204988e-01, 1.319684209290910e-01, 2.835508800145245e-02, 3.440943104661968e-01, 1.510708819611323e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_perdew_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_perdew", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.595001916762362e+01, 2.599766043420168e+01, 1.227900565878691e+01, 1.230045269070956e+01, 3.774798891934628e-01, 3.757875108360598e-01, 2.138709993645123e-01, -3.393218340734693e-01, 2.598325895271854e-02, -1.347400437018440e-01, -3.364378836784001e-01, -3.482345248385054e-01, -1.579228326555117e-01, -1.319974584419352e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_perdew_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_perdew", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.145629009473542e-03, 0.000000000000000e+00, 2.139839574560136e-03, 6.411055447865309e-03, 0.000000000000000e+00, 6.394713072898693e-03, 4.624209728193506e-01, 0.000000000000000e+00, 4.630114387200179e-01, 2.900255606658511e+00, 0.000000000000000e+00, 8.717347858781759e+03, 4.655507089994950e+01, 0.000000000000000e+00, 2.732588838335638e+08, 7.496737463447286e+03, 0.000000000000000e+00, 7.662719102582616e+03, 9.169738491151962e+08, 0.000000000000000e+00, 2.552205144323827e+09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
