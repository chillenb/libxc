
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_b86_mgc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86_mgc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.800487143588607e+00, -1.291055626236948e+00, -4.231409144388679e-01, -1.603705755715866e-01, -8.147788522270287e-02, -3.824158515255249e-02, -2.301608831449227e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_b86_mgc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86_mgc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.236721538629118e+00, -2.238855418670116e+00, -1.514212590119652e+00, -1.515576506260633e+00, -3.930994706933419e-01, -3.931942314154123e-01, -2.049312936209552e-01, -3.530487585876580e-02, -7.671361953844277e-02, -2.645765900499479e-03, -3.664799009203828e-02, -3.659338630308830e-02, -2.078349050847287e-03, -1.610804850186635e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_b86_mgc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86_mgc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.791795977553857e-04, 0.000000000000000e+00, -2.782268823379405e-04, -1.088026157469653e-03, 0.000000000000000e+00, -1.084571414666128e-03, -8.296748130364871e-02, 0.000000000000000e+00, -8.282272223474511e-02, -4.355343879821474e+00, 0.000000000000000e+00, -1.339200929992905e+02, -6.985622242833749e+01, 0.000000000000000e+00, -1.108268633614245e+06, -1.195105239141168e+02, 0.000000000000000e+00, -1.183147800993759e+02, -2.571061703106042e+06, 0.000000000000000e+00, -6.730740841075696e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
