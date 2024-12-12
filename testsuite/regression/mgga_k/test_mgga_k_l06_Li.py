
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_l06_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.956762996869940e+01, 8.234627000480570e+00, 8.408920370148953e-01, 1.345547876474417e-01, 2.959893977165264e-02, 1.535164291464752e-03, 5.454746473376860e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_l06_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([3.161240399359211e+01, 3.163732648738078e+01, 1.194563757352453e+01, 1.196711018055512e+01, 7.503211571174061e-01, 7.472101064753085e-01, 2.090255375326459e-01, 2.330517750800202e-03, 1.932607497576275e-02, 2.344190737395462e-06, 2.577078240913542e-03, 2.539724722985477e-03, 1.045854266539783e-06, 5.285654269429125e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_l06_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.163171474199523e-02, 0.000000000000000e+00, 1.161923495379766e-02, 8.052076022473717e-03, 0.000000000000000e+00, 8.030122084807596e-03, 2.979894505096132e-01, 0.000000000000000e+00, 3.026653387011220e-01, 3.675346698941400e+00, 0.000000000000000e+00, 1.210069199254532e-05, 6.924431626034671e+01, 0.000000000000000e+00, 1.372860173250363e-08, -6.388896605031148e-10, 0.000000000000000e+00, 1.402718832983906e-05, -2.974897035308784e-24, 0.000000000000000e+00, 1.553522104846671e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_l06_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_l06", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([-3.978740143395287e-03, -3.988138017249889e-03, -5.979082285018757e-03, -5.974488852222789e-03, 1.361241300917408e-03, 1.091006163619219e-03, -6.496582464516808e-03, 1.626127676264829e-11, -3.907711808196121e-03, -1.172939645042857e-19, 2.233177658649284e-15, 1.870666316382433e-11, 8.283546669583134e-35, -2.222930087887749e-21])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
