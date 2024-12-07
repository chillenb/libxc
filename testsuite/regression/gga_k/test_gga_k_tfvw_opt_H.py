
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_tfvw_opt_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tfvw_opt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.423842678142245e+00, 1.421072676402009e+00, 6.889024244144384e-01, 3.323713266404444e-01, 4.103758254404402e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_tfvw_opt_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tfvw_opt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.356697206314728e+00, 1.730992558736110e-16, 1.618384863767715e+00, 3.455086351174189e-16, 3.509061849853194e-01, 5.392996121794654e-17, -2.582125382946181e-01, 1.466893707359143e-16, -4.102112421708146e-01, -3.522826480034653e-17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_tfvw_opt_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_tfvw_opt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.511546610983475e-01, 0.000000000000000e+00, 0.000000000000000e+00, 3.483997682922377e-01, 0.000000000000000e+00, 0.000000000000000e+00, 1.741172764691432e+00, 0.000000000000000e+00, 0.000000000000000e+00, 9.141722640681736e+01, 0.000000000000000e+00, 0.000000000000000e+00, 8.743281936502757e+05, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
