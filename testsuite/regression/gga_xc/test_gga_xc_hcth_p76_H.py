
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hcth_p76_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_p76", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.954769215234978e-01, -4.972488180899637e-01, -2.612881400261998e-01, -1.150178823319349e-01, 1.412879207295791e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hcth_p76_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_p76", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.250813593391496e-01, -5.673201337444332e-02, -7.103302369564217e-01, -1.865469685363499e-02, -3.421927699098302e-01, 1.095733769247056e-03, -7.910069829809843e-02, 9.022396670462257e-02, -2.250787308051802e-04, -7.775407940037468e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hcth_p76_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_p76", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([6.123476338674017e-02, 0.000000000000000e+00, 9.053777453612817e+20, 8.306214723520493e-03, 0.000000000000000e+00, 4.207027189776889e+20, -5.951325565103457e-02, 0.000000000000000e+00, -5.005057916648119e+19, -9.238110275347509e+00, 0.000000000000000e+00, 4.370554078212469e+19, 4.545642683194805e+01, 0.000000000000000e+00, -5.984342117139681e+14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
