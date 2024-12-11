
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_xpbe_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_xpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.220587937960124e-01, -5.787391159035846e-01, -3.616957517974010e-01, -1.397991672974594e-01, -7.869894210256542e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_xpbe_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_xpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.282499047082879e-01, -1.296275516606161e-16, -7.170577067032790e-01, -2.209908376301310e-16, -3.984207828769879e-01, 2.257803763142829e-17, -1.384063802691549e-01, -6.894882876904170e-17, -1.048462352132594e-02, -9.789389146222695e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_xpbe_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_xpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.782081060195679e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.535857079696737e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.831014709884112e-01, 0.000000000000000e+00, 0.000000000000000e+00, -5.402034632347640e+00, 0.000000000000000e+00, 0.000000000000000e+00, -6.847206701435908e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
