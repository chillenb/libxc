
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_cc_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.996393927680305e-11, -1.665797812822538e-11, -8.986816739739807e-12, -8.100477194342690e-14, -9.182308123454235e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_cc_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.247092648353745e-02, -1.623546485025187e-01, -3.118079983775644e-02, -1.559040036724486e-01, -2.518518753837927e-02, -1.259259387238761e-01, -1.327196219147895e-02, -6.635981095942009e-02, -1.569790572941178e-03, -7.848953132844255e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cc_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.328164555807767e+00, 2.656329111615534e+00, 1.328164555807767e+00, 3.862177937256196e-02, 7.724355874512391e-02, 3.862177937256196e-02, 1.466740715437317e-01, 2.933481430874634e-01, 1.466740715437317e-01, 3.983710525481820e+00, 7.967421050963640e+00, 3.983710525481820e+00, 3.345027873366872e+03, 6.690055746733743e+03, 3.345027873366872e+03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cc_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-3.167652012610543e+00, -3.166055751490881e+00, -6.640201267987264e-02, -6.595706324631133e-02, -5.045896111229682e-02, -5.035756165266289e-02, -2.610276748218711e-02, -2.610261112728963e-02, -2.291670005018553e-03, -2.291670018167617e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
