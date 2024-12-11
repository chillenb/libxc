
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_sol_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_sol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.703036616384539e-02, -5.108202438662889e-02, -5.363725291271257e-03, -1.600564332618745e-02, -2.216562647125218e-03, -1.600309951395318e-08, -3.818017950407704e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_sol_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_sol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.173244393885396e-01, -1.171889567566819e-01, -1.054179946733774e-01, -1.053125123607990e-01, -2.548144775125539e-02, -2.549076714844199e-02, -2.345038003566964e-02, -1.061228392576142e-01, -9.657544369611720e-03, 4.670283521064983e-01, -1.035495587877013e-07, -1.040697625306304e-07, -2.393925173111572e-15, -2.835087037095924e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_sol_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe_sol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.028547001501965e-05, 8.057094003003931e-05, 4.028547001501965e-05, 1.373922762266883e-04, 2.747845524533765e-04, 1.373922762266883e-04, 4.882146493292004e-03, 9.764292986584010e-03, 4.882146493292004e-03, 2.489468031800216e+00, 4.978936063600431e+00, 2.489468031800216e+00, 1.594123656228095e+01, 3.188247312456189e+01, 1.594123656228095e+01, 3.529155396068316e-04, 7.058310792173810e-04, 3.529155396068316e-04, 3.380088498550873e-06, 6.759926358451161e-06, 3.380088498550873e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
