
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_mohlyp2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mohlyp2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.536440667525502e-01, -5.899572021430666e-01, -3.600834384271150e-01, -1.852986697393579e-01, -1.245654106107754e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_mohlyp2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mohlyp2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.715174372642537e-01, -1.180856916259592e-01, -7.671842976338346e-01, -1.249547687509144e-01, -4.029842101361285e-01, -9.833348431135434e-02, -1.150649755718841e-01, -1.777544569060721e-02, -1.658307218880388e-02, -1.226769921899730e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_mohlyp2_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_mohlyp2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.224806089010646e-04, 1.462410058667360e-02, 1.096387466450042e-02, -9.022858501201669e-03, 2.323458825295151e-02, 1.740332539156102e-02, -1.684403351914133e-01, 1.993385757292538e-01, 1.494995148087814e-01, -1.485790456356596e+01, 8.513935189140986e+00, 6.385440588210187e+00, -2.049577812152807e+01, 3.578904936287713e-18, 2.684174687662640e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
