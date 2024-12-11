
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_pjs18_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_pjs18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.696761867081864e+00, -1.088726782607713e+00, -5.028826422975879e-02, -4.533774732274014e-02, -3.815450776700051e-03, -1.897167097000193e-03, -4.020403785185744e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_pjs18_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_pjs18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.405471107133630e+00, -2.407907606928768e+00, -1.622791789435073e+00, -1.624286121177686e+00, -2.441441999108932e-01, -2.441993154933640e-01, -8.099076361935914e-02, -6.291471206129785e-06, -7.627347743433000e-03, -8.757029977911763e-10, 7.605247051401049e-04, -6.509191303997044e-06, 2.172445081875864e-05, -9.499414337933800e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_pjs18_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_pjs18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.728558845518112e-04, 0.000000000000000e+00, -2.717953344488158e-04, -1.330760542746797e-03, 0.000000000000000e+00, -1.325385299457587e-03, -7.024901349935581e-01, 0.000000000000000e+00, -6.882701859062857e-01, -2.457788522833830e-01, 0.000000000000000e+00, -3.390147726054458e+00, -5.769215714887098e-01, 0.000000000000000e+00, -3.947014622257575e+00, -1.998354146730686e+00, 0.000000000000000e+00, -3.358158242075673e+00, -3.511465615820493e+00, 0.000000000000000e+00, -5.278921377063954e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_pjs18_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_pjs18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.198759867254363e-02, 1.198130775838956e-02, 1.672298956851676e-02, 1.671048522244789e-02, 3.358949716647960e-02, 3.358553404570646e-02, 1.941242768270869e-02, 1.352961758117568e-05, 2.462207871730423e-03, 4.901867144367799e-10, 9.910872906023197e-06, 1.526851349137438e-05, 1.320598958794626e-10, 5.255904237818155e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
