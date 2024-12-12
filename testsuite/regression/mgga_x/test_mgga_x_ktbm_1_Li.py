
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.995784615742025e+00, -1.407355976107956e+00, -3.426077137188496e-01, -1.788287503914206e-01, -7.490342288707033e-02, -1.280978805143287e-02, -2.354295025587896e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.486846227160526e+00, -2.489272723694343e+00, -1.660912627111923e+00, -1.662375935186644e+00, -4.237811714105936e-01, -4.252089179058127e-01, -2.294470057936404e-01, -1.626841118127995e-02, -8.811200452480231e-02, -5.159640543147227e-04, -1.669601045190515e-02, -1.698290704332911e-02, -3.362265508809423e-04, -2.450035380670476e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.329093940555413e-04, 0.000000000000000e+00, -5.310815419041241e-04, -2.113017312182568e-03, 0.000000000000000e+00, -2.106915450143233e-03, -5.556855638205195e-02, 0.000000000000000e+00, -5.739986557240634e-02, -8.149106501112062e+00, 0.000000000000000e+00, -1.158135219851722e+01, -7.952892436117202e+01, 0.000000000000000e+00, -2.894646072396701e+04, -3.908612784629282e-01, 0.000000000000000e+00, -1.035643173946059e+01, -8.048958977965175e-01, 0.000000000000000e+00, -1.310446505116527e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_1_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.961935159466616e-02, 1.960429260953831e-02, 2.724788212780821e-02, 2.724113903245748e-02, 1.496288647897850e-02, 1.604068004424826e-02, 2.185644560421467e-01, 1.481300915887135e-04, 2.058466041419199e-01, 1.179385700488004e-05, 1.184402215527222e-07, 1.507074009326763e-04, 7.738381255557559e-16, 5.716590176546620e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
