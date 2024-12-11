
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_apbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.644324610729039e+01, 8.184139970376739e+00, 6.411526455121976e-01, 1.324207897192007e-01, 2.669015056751176e-02, 1.232370275442637e-03, 4.381281427819385e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_apbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.591222156887400e+01, 2.595970072895550e+01, 1.232398022933002e+01, 1.234527538214560e+01, 8.363720758998703e-01, 8.363330459527832e-01, 2.135071176809606e-01, 1.869413928103568e-03, 3.413581901875931e-02, 1.882861027244416e-06, 2.066875476720363e-03, 2.037062799627099e-03, 8.400350317643425e-07, 4.245464345773928e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_apbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_apbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.517814341050006e-03, 0.000000000000000e+00, 2.511435022751708e-03, 6.875752993507863e-03, 0.000000000000000e+00, 6.859629465482597e-03, 1.127248711992743e-01, 0.000000000000000e+00, 1.123808445045585e-01, 3.535992530686697e+00, 0.000000000000000e+00, 1.461173466473886e-02, 2.264333616370724e+01, 0.000000000000000e+00, 2.963715438733082e-03, 1.561470982888272e-02, 0.000000000000000e+00, 1.447517866916015e-02, 1.441069290015871e-03, 0.000000000000000e+00, 1.466421291736263e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
