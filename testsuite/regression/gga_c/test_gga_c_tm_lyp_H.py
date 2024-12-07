
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_tm_lyp_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tm_lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-8.779723354083784e-15, -1.302018265365277e-14, -5.002603365922616e-14, -5.553027931880848e-13, -5.240125451836643e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_tm_lyp_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tm_lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.398135497411436e-15, -2.620581848413338e-01, 1.410041357361179e-15, -2.798593877388815e-01, 7.254118380734093e-15, -2.151157889327011e-01, -5.343297451234935e-13, -4.548190501811585e-02, -1.696846832321122e-10, -4.487495782171297e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_tm_lyp_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tm_lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.080677839108742e-16, 3.656290438134858e-02, 2.741167558519927e-02, -1.896029939795327e-15, 5.756090553319395e-02, 4.311465122261530e-02, -5.371822627040478e-14, 4.529443752754222e-01, 3.396982450151208e-01, 5.991985014168422e-11, 7.369060291395690e+00, 5.526785867649750e+00, 2.681414892675919e-38, 7.571175106522357e-33, 5.678366946635025e-33]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
