
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lcy_pbe_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.055908556670653e-01, -3.443411612444963e-01, -1.487366287753885e-01, -7.963452675965144e-03, -9.564755895635265e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lcy_pbe_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-5.917247541736548e-01, 1.596983044605467e+00, -5.099905867822331e-01, 7.387469275550377e+01, -2.343076268837189e-01, 4.139501778593020e+01, -1.549998460735452e-02, 3.382974671448072e-01, -1.913778286316104e-06, 1.066995416500189e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lcy_pbe_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [8.647171387043002e-03, 3.288056552649452e-02, 1.644028276324726e-02, 1.903154052221023e-04, 2.041382516785488e-02, 1.020691258392744e-02, 2.176288633567600e-03, 8.417394271524091e-02, 4.208697135762044e-02, 5.100445946788960e-02, 1.762212799391784e-01, 8.811063996958873e-02, 9.924762089410135e-04, 1.985451945004567e-03, 9.927259729551347e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
