
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_gam_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_gam", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.946290780393604e+00, -1.351315737322114e+00, -4.372124873461359e-01, -1.832397330986335e-01, -7.019184116526883e-02, -5.648189427750568e-02, -1.079832553922620e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_gam_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_gam", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.672499694784970e+00, -2.675100027253914e+00, -1.778045355686204e+00, -1.779752901839821e+00, -1.192718848996661e-01, -1.178895166708977e-01, -2.386297879113820e-01, -7.106275610958734e-02, -9.700940762148084e-02, -2.332820012457109e-03, -7.454715737260705e-02, -7.406565720442848e-02, -1.558662561343970e-03, -1.108255323431368e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_gam_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_gam", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.405108994212531e-04, 0.000000000000000e+00, 1.404223012999796e-04, -5.436492512464040e-05, 0.000000000000000e+00, -5.303716607513977e-05, -2.397082136836798e-01, 0.000000000000000e+00, -2.403005826547243e-01, -1.000831232914897e+00, 0.000000000000000e+00, -5.846791014619535e+00, 1.029833615815916e+01, 0.000000000000000e+00, -3.775217487464026e+01, -5.937364315616301e+00, 0.000000000000000e+00, -5.545834964888691e+00, -2.748419400354927e+01, 0.000000000000000e+00, -3.934238277496671e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
