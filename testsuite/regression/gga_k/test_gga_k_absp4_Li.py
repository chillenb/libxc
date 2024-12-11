
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_absp4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_absp4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([8.612125010569411e+00, 5.037578679306975e+00, 1.567995928245705e+00, 6.462599518337857e-02, 3.946668186338905e-02, 1.322965651464824e+00, 5.818600397543918e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_absp4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_absp4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([9.520685040559535e+00, 9.541149727071105e+00, 3.703515078881748e+00, 3.711885716645428e+00, -1.061748934741185e+00, -1.066128049877525e+00, 8.406104169725279e-02, -1.310475126273318e+00, -1.616533620703313e-02, -5.189649908314732e-01, -1.299743110950561e+00, -1.345121843019742e+00, -6.082532685825074e-01, -5.083989504212026e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_absp4_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_absp4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([8.264121882859804e-03, 0.000000000000000e+00, 8.241823487985482e-03, 2.469271782793146e-02, 0.000000000000000e+00, 2.462977392501605e-02, 1.781049200317202e+00, 0.000000000000000e+00, 1.783323424023849e+00, 1.117080829737512e+01, 0.000000000000000e+00, 3.357552131977851e+04, 1.793103690721600e+02, 0.000000000000000e+00, 1.052477155736057e+09, 2.887424852263887e+04, 0.000000000000000e+00, 2.951353929705318e+04, 3.531793788592522e+09, 0.000000000000000e+00, 9.830010184733633e+09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
