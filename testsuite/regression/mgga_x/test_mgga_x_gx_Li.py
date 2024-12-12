
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_gx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.140022595335194e+00, -1.485698365893895e+00, -3.508474184869329e-01, -1.931668870998599e-01, -7.669328582114858e-02, -1.184576525891439e-02, -1.857135236633666e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_gx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.763248042770171e+00, -2.765885783724683e+00, -1.855840686067126e+00, -1.857518098050420e+00, -6.555324431359302e-02, -6.843882476092064e-02, -2.533488115619998e-01, 1.026153714786147e+01, -6.014023052557348e-02, 1.288534107863708e+02, -1.299167788396100e-02, 1.008882393225258e+01, -2.617191072232138e-04, 4.849958628171022e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.061170242356274e-04, 0.000000000000000e+00, -4.046524574013690e-04, -1.747833013008836e-03, 0.000000000000000e+00, -1.741800204696412e-03, -5.358715620599420e-01, 0.000000000000000e+00, -5.256434479597557e-01, -6.073327575720131e+00, 0.000000000000000e+00, -2.632781328878477e+05, -2.457678095319775e+02, 0.000000000000000e+00, -2.613196544555025e+11, -7.438080478904114e-06, 0.000000000000000e+00, -2.216900296823392e+05, -4.890598958185961e-12, 0.000000000000000e+00, -9.377548082821584e+11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gx_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.107297905427289e-02, 2.105379194513418e-02, 3.035306125412128e-02, 3.032559744526879e-02, 1.290197652751474e-01, 1.263957860834246e-01, 2.331382655494147e-01, 3.362509180130806e+00, 5.877478487225314e-01, 1.064708817317555e+02, 1.104642216543328e-10, 3.221041510522955e+00, 5.938169401791733e-22, 4.090786233411168e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
