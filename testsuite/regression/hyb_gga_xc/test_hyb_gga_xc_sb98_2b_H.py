
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_sb98_2b_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.860253571058649e-01, -4.535891631809945e-01, -2.771803797650224e-01, -1.198230627643792e-01, -8.673125816432436e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_sb98_2b_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.428595219312365e-01, -2.526683212743771e-01, -5.847041095928938e-01, -2.467973793699304e-01, -3.331534665208342e-01, -1.962021101737743e-01, -8.370398999871984e-02, 8.480828191409639e-03, -1.149387853283071e-02, 1.961506891120325e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_sb98_2b_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_2b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.695711325234708e-02, 0.000000000000000e+00, -1.398240538903062e+20, -7.475116766421838e-03, 0.000000000000000e+00, -8.405024605457056e+19, -7.319964184432799e-02, 0.000000000000000e+00, -1.996338337694617e+18, -8.438001284242397e+00, 0.000000000000000e+00, 4.601535481656308e+19, -1.612954302637261e+01, 0.000000000000000e+00, 5.319085099438075e+13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
