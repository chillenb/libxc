
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_th4_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.500237202863347e-01, -6.131059489805250e-01, -3.752553767856934e-01, -9.511371972346999e-02, 6.107211882781337e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_th4_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.778438339336978e-01, -1.648017966964986e-01, -7.397895570192199e-01, -5.944478732186907e-02, -4.348236704192873e-01, 7.856023711118265e-03, -1.429068633740418e-01, -6.512963891634255e-02, -8.868925555945506e-02, -2.630225699678002e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_th4_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.095088509446008e-03, 5.693522399830251e-03, 2.567150289345933e+17, -4.379746333020135e-02, -4.168149214751457e-03, 1.479812249635882e+17, -1.574100581501768e-01, -3.163672965064512e-01, -3.153588426620329e+17, 1.013818437417203e+00, -4.827417373953004e+01, -9.592689133843242e+17, 1.544779407696570e+05, -7.807682232090170e+05, -9.646259568953500e+17]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
