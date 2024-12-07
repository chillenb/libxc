
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_task_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.037370272529753e+00, -1.412916089498220e+00, -3.223283518957000e-01, -1.836632824628109e-01, -7.184050300756316e-02, -6.633564992211092e-03, -2.726474506980853e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_task_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.515146251623531e+00, -2.517431149965953e+00, -1.597867115935475e+00, -1.893635624781331e+00, -2.312141191905988e-01, -4.634036458143668e-01, -2.362335095557474e-01, -1.124692921511024e-02, 1.567045258482484e-02, -1.097090100564590e-04, -1.193703391975470e-02, -1.184713620106153e-02, -5.924996593064389e-05, -3.745105620817320e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_task_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.270703498750685e-04, 0.000000000000000e+00, -9.251332095094493e-04, -4.089317971260953e-03, 0.000000000000000e+00, 4.561096218325003e-05, -2.756990806134532e-01, 0.000000000000000e+00, 1.227502019999721e-02, -1.252455671181048e+01, 0.000000000000000e+00, 2.805042303557318e+01, -6.624073170168936e+02, 0.000000000000000e+00, 2.486413190074975e+04, 2.525265818399668e+01, 0.000000000000000e+00, 2.519790168521046e+01, 4.130692887391659e+04, 0.000000000000000e+00, 8.679127117842814e+04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_task_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_task_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_task", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [4.848227197777395e-02, 4.850994886677103e-02, 7.181693873001560e-02, 7.233440849760966e-12, 6.868695537056696e-02, 3.249572850000783e-11, 4.839333002674311e-01, 1.101738966237264e-12, 1.601376819552769e+00, 7.168610182131681e-07, 2.580324382400690e-13, 1.171820848926183e-12, 4.422969922432040e-31, 2.848356001364977e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
