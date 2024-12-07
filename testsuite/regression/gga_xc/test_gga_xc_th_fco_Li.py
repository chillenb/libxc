
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_th_fco_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fco", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.155760114255701e+00, -1.521679921089201e+00, -6.641514648956015e-01, -3.117433932063836e-01, -2.053520502499509e-01, -3.594890223709508e-01, 4.503815673602923e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_th_fco_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fco", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.960449619179383e+00, -2.954785951007058e+00, -1.977708296967176e+00, -1.976871348023827e+00, -4.486853552785777e-01, -4.484907066242122e-01, -3.648016448784924e-01, -2.511092279902005e-01, -1.830267038906897e-01, -1.795075311802878e-01, -2.174808325740222e-01, -2.154748650777680e-01, -7.986241083767502e-01, -1.167582817586588e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_th_fco_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th_fco", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.615529017884899e-04, -5.413354717164238e-04, 4.601823303322319e-04, 2.221382622209339e-04, -4.364844370035893e-04, 2.185290717275167e-04, -1.498022768701398e-01, -5.438594702417244e-02, -1.497946881295233e-01, -1.072696457631029e+01, -1.829503435424898e+00, -2.164139406266232e+03, -1.462141784632917e+02, -3.100954109190608e+01, -1.835604766277194e+08, -3.518235189042328e+03, 2.832760311436229e+03, -3.535614131720906e+03, 2.537507333802030e+07, 1.497887721368609e+10, -1.677206946150691e+09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
