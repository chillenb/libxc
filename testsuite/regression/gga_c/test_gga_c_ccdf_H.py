
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_ccdf_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_ccdf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.524064685582033e-02, -2.700720505793804e-02, -2.002685704129811e-02, -1.712986261157160e-02, -3.432091821024318e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_ccdf_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_ccdf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.574621569018082e-02, -4.574621569018082e-02, -1.049417703303197e-01, -1.049417703303197e-01, -2.045981833750817e-02, -2.045981833750817e-02, -1.825654294423187e-02, -1.825654294423187e-02, -4.392135975354268e-03, -4.392135975354268e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_ccdf_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_ccdf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([5.594169483560285e-05, 1.118833896712057e-04, 5.594169483560285e-05, 3.604519535918432e-02, 7.209039071836863e-02, 3.604519535918432e-02, 4.741421559097695e-05, 9.482843118195389e-05, 4.741421559097695e-05, 7.238985619304584e-29, 1.447797123860917e-28, 7.238985619304584e-29, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
