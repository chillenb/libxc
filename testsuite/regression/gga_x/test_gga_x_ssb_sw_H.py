
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ssb_sw_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ssb_sw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.536625102940711e-01, -5.882032452093345e-01, -3.575843102391632e-01, -1.343105622520069e-01, -7.396560382427567e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ssb_sw_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ssb_sw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.714975043529798e-01, -1.374344732032296e-16, -7.717542038977556e-01, -1.503672133222701e-16, -4.049268823499296e-01, 4.329743544846822e-19, -1.385626678097244e-01, -6.314658941681554e-17, -9.855017437925455e-03, -8.648253338394296e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ssb_sw_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ssb_sw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.054253644400394e-04, 0.000000000000000e+00, 0.000000000000000e+00, -5.813922673804987e-03, 0.000000000000000e+00, 0.000000000000000e+00, -1.569204281341260e-01, 0.000000000000000e+00, 0.000000000000000e+00, -4.560713409264804e+00, 0.000000000000000e+00, 0.000000000000000e+00, -5.644039078845589e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
