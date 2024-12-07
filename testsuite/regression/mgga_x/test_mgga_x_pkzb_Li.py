
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_pkzb_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.804562442512226e+00, -1.284005167778990e+00, -4.422533911499442e-01, -1.613992902488042e-01, -8.254125808473214e-02, -2.055685952400989e-02, -3.838588870219948e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_pkzb_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.273239682166834e+00, -2.275421204679987e+00, -1.523355787452314e+00, -1.524749751857006e+00, -3.260298533746764e-01, -3.304316126644019e-01, -2.084726917842723e-01, -2.615897632319181e-02, -5.890393252347759e-02, -8.296468243197437e-04, -2.750809932721536e-02, -2.730785489448631e-02, -5.541564195188991e-04, -3.939545845215671e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_pkzb_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.679060992774058e-04, 0.000000000000000e+00, -2.669351543375862e-04, -1.188604551525488e-03, 0.000000000000000e+00, -1.184701158800375e-03, -1.670717720197593e-01, 0.000000000000000e+00, -1.643796131439423e-01, -3.929036071419318e+00, 0.000000000000000e+00, -1.624110715563185e-03, -1.426125264847406e+02, 0.000000000000000e+00, -2.638442325255194e-05, 1.957353455408861e-08, 0.000000000000000e+00, -1.636325606617371e-03, 4.056546354834368e-21, 0.000000000000000e+00, -6.327853485848266e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_pkzb_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_pkzb_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.488605137678329e-03, 3.485145028346126e-03, 5.505991087997634e-03, 5.502694026426871e-03, 1.485618880366460e-02, 1.476161183081960e-02, 3.657208603725504e-02, 8.678093643874245e-09, 1.174004764944084e-01, 4.498396315174000e-15, -1.644485056706652e-13, 9.946590835096205e-09, -2.770486294585053e-31, 1.155115432031189e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
