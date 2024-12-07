
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbe_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.206369600616237e-02, -1.817876198856552e-02, -8.087390118922740e-03, -1.586019502253216e-04, -2.329818642697601e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbe_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.697773452122462e-02, 1.596983044605467e+00, -4.114829558466256e-02, 7.387469275550377e+01, -2.759497946421170e-02, 4.139501778593020e+01, -9.425047225492048e-04, 3.382974671448072e-01, -1.515817973311157e-09, 1.066995416500183e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbe_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.644028276324726e-02, 3.288056552649452e-02, 1.644028276324726e-02, 1.020691258392744e-02, 2.041382516785488e-02, 1.020691258392744e-02, 4.208697135762044e-02, 8.417394271524091e-02, 4.208697135762044e-02, 8.811063996958873e-02, 1.762212799391784e-01, 8.811063996958873e-02, 9.927259729551347e-04, 1.985451945004567e-03, 9.927259729551347e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
