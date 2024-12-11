
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_kt2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.022507024236239e+00, -1.487100096135634e+00, -3.812506337518011e-01, -1.786170353889594e-01, -7.302026213010067e-02, -1.613412344978359e-02, -3.220087327511423e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_kt2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.396925532996109e+00, -2.399189197747294e+00, -1.586466747961318e+00, -1.587868921605542e+00, -4.415480478109925e-01, -4.413951591259060e-01, -2.363716457517639e-01, -8.990596045351848e-02, -9.646895262677908e-02, -4.339231598176022e-02, -2.127095394955824e-02, -2.120660229559303e-02, -4.396141382998249e-04, -3.974077742015946e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_kt2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_kt2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.919594560353022e-04, 0.000000000000000e+00, -4.902048214802675e-04, -2.061349402651428e-03, 0.000000000000000e+00, -2.054586108301680e-03, -5.486396554400828e-02, 0.000000000000000e+00, -5.487195057375636e-02, -5.951829666719937e-02, 0.000000000000000e+00, -5.999998880484111e-02, -5.998800963135268e-02, 0.000000000000000e+00, -5.999999992242971e-02, -5.999998631077526e-02, 0.000000000000000e+00, -5.999998670470278e-02, -5.999999999999771e-02, 0.000000000000000e+00, -5.999999999999940e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
