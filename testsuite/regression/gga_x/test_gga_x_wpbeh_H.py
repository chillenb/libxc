
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_wpbeh_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_wpbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.219439490983338e-01, -5.760817219329533e-01, -3.608657973405664e-01, -1.342887737227818e-01, -7.399215141239782e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_wpbeh_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_wpbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-8.283994112906186e-01, -1.267399301841831e-16, -7.144741322867403e-01, 5.264173200651784e-18, -3.962717176144442e-01, -1.084059704500323e-16, -1.409416598787046e-01, -8.851615390987892e-18, -9.865621340345070e-03, -1.149107417309280e-18]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_wpbeh_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_wpbeh", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.317881701484391e-02, 0.000000000000000e+00, 0.000000000000000e+00, -2.491283811902139e-02, 0.000000000000000e+00, 0.000000000000000e+00, -1.853781329063748e-01, 0.000000000000000e+00, 0.000000000000000e+00, -4.289664151756575e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
