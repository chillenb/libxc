
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_hjs_pbe_sol_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_pbe_sol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.707734201957082e+00, -1.189223367586742e+00, -3.214798575516389e-01, -1.047901905109490e-01, -2.588263043172699e-02, -3.382975786530186e-04, -1.636588659304272e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_hjs_pbe_sol_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_pbe_sol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.210104060223008e+00, -2.212222019353661e+00, -1.490111069834367e+00, -1.491486308638431e+00, -3.003128303724102e-01, -3.003388039283367e-01, -1.516889881915956e-01, -6.347523112499359e-04, -3.110517932696936e-02, -1.322334348149775e-08, -7.698296188094804e-04, -7.486573852206828e-04, -3.940525674586694e-09, -1.415774469951284e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_hjs_pbe_sol_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_hjs_pbe_sol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.478397088529814e-04, 0.000000000000000e+00, -1.473206128258398e-04, -6.070885858742765e-04, 0.000000000000000e+00, -6.050937265116718e-04, -7.196314271816487e-02, 0.000000000000000e+00, -7.186100103108073e-02, -1.534449161579154e+00, 0.000000000000000e+00, -1.388767443364109e-02, -2.905583515518401e+01, 0.000000000000000e+00, -2.002333038187901e-09, -1.750089719818494e-02, 0.000000000000000e+00, -1.585117384593514e-02, -2.717381788087456e-10, 0.000000000000000e+00, -9.666890083598439e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
