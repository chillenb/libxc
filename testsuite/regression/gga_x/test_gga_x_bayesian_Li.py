
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_bayesian_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bayesian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.786604577253229e+00, -1.275936612209188e+00, -4.301262688327925e-01, -1.595685687618827e-01, -8.098882973850086e-02, -3.354005987070044e-02, -6.564850931614244e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_bayesian_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bayesian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.245936853067301e+00, -2.248086714384204e+00, -1.515397108241541e+00, -1.516777505432928e+00, -3.531316331714695e-01, -3.531703902042404e-01, -2.058213299048733e-01, -4.081777243366592e-02, -7.184652904639736e-02, -1.414100609696174e-03, -4.270481115558300e-02, -4.249271605902676e-02, -9.462700791337159e-04, -6.731534871519465e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_bayesian_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bayesian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.316619263267377e-04, 0.000000000000000e+00, -2.308237880142247e-04, -9.755955008077588e-04, 0.000000000000000e+00, -9.723757613497997e-04, -1.068502687280307e-01, 0.000000000000000e+00, -1.067357697446562e-01, -3.402822310974209e+00, 0.000000000000000e+00, -1.869950292417814e+01, -7.907966066704232e+01, 0.000000000000000e+00, -2.576926487846498e+03, -1.791324073936544e+01, 0.000000000000000e+00, -1.717832992140464e+01, -3.045794220260709e+03, 0.000000000000000e+00, -5.610192679706359e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
