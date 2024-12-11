
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbeloc_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbeloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.223726131975330e-02, -1.769147803041506e-02, -4.117853623090181e-03, -5.301681702858316e-06, -1.033353090595934e-14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbeloc_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbeloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.676418777347409e-02, 2.127043311962064e-01, -5.160098850728106e-02, 2.081843027777553e+01, -2.654515719525006e-02, 7.408157211478209e+00, -5.529088575856843e-05, 1.314762321038589e-04, -1.152363794665323e-13, -4.565544990675630e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbeloc_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbeloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([9.628541982524549e-03, 1.925708396504910e-02, 9.628541982524549e-03, 1.524870455071232e-02, 3.049740910142464e-02, 1.524870455071232e-02, 5.083036130800587e-02, 1.016607226160117e-01, 5.083036130800587e-02, 5.982249910496439e-03, 1.196449982099288e-02, 5.982249910496439e-03, 8.792121240540160e-08, 1.758424248108032e-07, 8.792121240540160e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
