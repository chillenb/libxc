
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_beefvdw_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_beefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.428347593149224e-01, -5.925198467855485e-01, -3.716945779829164e-01, -1.475196160139048e-01, -7.667494449940930e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_beefvdw_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_beefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-8.562914590378692e-01, -1.105268417703541e-16, -7.409664992931636e-01, -2.305444053491548e-16, -3.950099643221022e-01, -5.524319498081288e-17, -1.657013407490904e-01, -8.444619965365642e-17, -1.021872721690650e-02, -3.576699175095517e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_beefvdw_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_beefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.260155860179892e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.278785841269021e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.196661685690042e-01, 0.000000000000000e+00, 0.000000000000000e+00, -3.488399614969973e+00, 0.000000000000000e+00, 0.000000000000000e+00, -3.674829493268318e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
