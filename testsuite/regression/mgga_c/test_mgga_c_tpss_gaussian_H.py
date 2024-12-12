
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_tpss_gaussian_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.867375404581885e-11, -1.942267136831315e-11, -5.770000257568691e-12, -3.868901486025145e-16, -5.496137701394224e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_tpss_gaussian_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-6.412737057909071e-02, -6.412804555982708e-02, -3.635708767062761e-02, -3.635630577962624e-02, -1.617439958134836e-02, -1.617410412925521e-02, -3.171895542918831e-04, -3.171895462392698e-04, -4.663610551879712e-10, -4.663559431673045e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpss_gaussian_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([2.623014181658332e+00, 5.246028363316665e+00, 2.623014181658332e+00, 4.503333543194678e-02, 9.006667086389356e-02, 4.503333543194678e-02, 9.419683838328019e-02, 1.883936767665605e-01, 9.419683838328019e-02, 9.520757727997654e-02, 1.904151545375711e-01, -4.629789078041842e-01, 9.937571945837971e-04, 1.987514391641490e-03, 1.041843935662619e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpss_gaussian_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-6.255848430183772e+00, -6.182507903921132e+00, -7.742533249955175e-02, -7.616321235615631e-02, -3.240569074917187e-02, -3.228794661363517e-02, -6.238358024207600e-04, -6.238196417223381e-04, -6.808204857017469e-10, -6.808204910097920e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
