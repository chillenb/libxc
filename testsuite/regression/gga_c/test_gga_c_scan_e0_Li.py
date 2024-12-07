
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_scan_e0_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_scan_e0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.441460042978385e-02, -5.129435772149726e-02, -1.900591908694322e-02, -1.598329399270094e-02, -5.988253762813859e-03, -9.017963485753629e-04, -4.365282159823870e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_scan_e0_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_scan_e0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.082654178349468e-01, -1.081366457964615e-01, -9.146442738522818e-02, -9.136395657440877e-02, -3.564605339719529e-02, -3.566613665421209e-02, -2.264444964107708e-02, -1.101584852481693e-01, -1.090126252468267e-02, 7.493492276359834e-03, -1.705616427238801e-03, -1.720476637893796e-03, -7.940925234354263e-06, -1.067555803108795e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_scan_e0_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_scan_e0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [3.508961566231346e-05, 7.017923132462693e-05, 3.508961566231346e-05, 9.997931067703667e-05, 1.999586213540733e-04, 9.997931067703667e-05, 3.774589170511780e-03, 7.549178341023559e-03, 3.774589170511780e-03, 2.140853172337111e+00, 4.281706344674221e+00, 2.140853172337111e+00, 8.957110947398082e+00, 1.791422189479616e+01, 8.957110947398082e+00, 2.523218757907926e+00, 5.046437515815851e+00, 2.523218757907926e+00, 4.880908788100187e+03, 9.761817576200374e+03, 4.880908788100187e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
