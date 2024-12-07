
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_bkl2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bkl2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.764420970400042e+00, -1.244513558228558e+00, -3.872032759935986e-01, -1.583965992088430e-01, -7.396156117999916e-02, -1.140066209733568e-02, -2.127820881496785e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_bkl2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bkl2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.276931865720625e+00, -2.279048428807669e+00, -1.556579384860534e+00, -1.557958769401790e+00, -3.275243540034232e-01, -3.274713972988623e-01, -2.072975590657403e-01, -1.456202245446447e-02, -7.169344831927725e-02, -4.598929181543383e-04, -1.538194560108184e-02, -1.523417104160078e-02, -3.071820507310970e-04, -2.183783727950879e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_bkl2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_bkl2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.278391883529793e-04, 0.000000000000000e+00, -1.273846378843228e-04, -5.378705058211114e-04, 0.000000000000000e+00, -5.360713584928972e-04, -9.151968642400192e-02, 0.000000000000000e+00, -9.145820855789265e-02, -1.928492671235018e+00, 0.000000000000000e+00, 5.551949404334336e-01, -5.891156752118867e+01, 0.000000000000000e+00, 0.000000000000000e+00, 1.040102080488469e+00, 0.000000000000000e+00, 7.459756743492668e-01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
