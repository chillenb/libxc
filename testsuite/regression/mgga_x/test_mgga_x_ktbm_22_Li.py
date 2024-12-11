
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_22_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.005424854565952e+00, -1.313186018272687e+00, -2.471281201587778e-01, -1.853913734748887e-01, -5.336725766904249e-02, -9.934649235458415e-03, -1.863315350341092e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_22_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.868204644163184e+00, -2.870991418497173e+00, -1.994422248579573e+00, -1.996190854002201e+00, -3.225074060364069e-01, -3.225311840982349e-01, -2.582863558633502e-01, -1.226468745946186e-02, -7.493128763243907e-02, -3.889113153308937e-04, -1.289494368386001e-02, -1.280348924668040e-02, -2.597703019610545e-04, -1.945956565401307e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_22_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.031523427039715e-04, 0.000000000000000e+00, -8.004603187701551e-04, -2.899932777535740e-03, 0.000000000000000e+00, -2.892716237493706e-03, -2.471688862042922e-02, 0.000000000000000e+00, -2.645603783926081e-02, -1.275059220016835e+01, 0.000000000000000e+00, -9.752721938390247e+00, -6.145297662242157e+01, 0.000000000000000e+00, -2.434501761868927e+04, -1.807642848588552e-01, 0.000000000000000e+00, -8.721891201120027e+00, -3.686499498244624e-01, 0.000000000000000e+00, 2.485681393190915e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_22_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [2.789706149897243e-02, 2.786673124743080e-02, 3.574335973875774e-02, 3.574116818843884e-02, -1.296154843781280e-03, -1.273335309569719e-03, 3.135075454964014e-01, 1.250430128667067e-04, 3.160738818816392e-02, 9.919120862744269e-06, 2.684800922961186e-06, 1.272473689420676e-04, 4.476002234155036e-11, -7.513030077163100e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
