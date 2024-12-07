
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_mpbe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.794809627002782e+00, -1.284682103005663e+00, -4.249634377004232e-01, -1.600378725627289e-01, -8.152084509478376e-02, -2.038799850792803e-02, -3.807966465657566e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_mpbe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.241030281818575e+00, -2.243167913001081e+00, -1.515173547316438e+00, -1.516546928940255e+00, -3.983833497193248e-01, -3.986262652495288e-01, -2.052585788429945e-01, -2.593327352403289e-02, -7.472642372086306e-02, -8.230270912873638e-04, -2.726850658871727e-02, -2.707114366770563e-02, -5.497353857967395e-04, -3.908117315286539e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_mpbe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_mpbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.588206886846756e-04, 0.000000000000000e+00, -2.579229072694642e-04, -1.038197594065822e-03, 0.000000000000000e+00, -1.034837604233884e-03, -8.158479401583568e-02, 0.000000000000000e+00, -8.136743164347994e-02, -3.983943957604796e+00, 0.000000000000000e+00, -1.101397724971588e-01, -7.433001238775333e+01, 0.000000000000000e+00, -6.957519671960894e-01, -1.120852371212746e-01, 0.000000000000000e+00, -1.045987650427733e-01, -5.064716633771361e-01, 0.000000000000000e+00, -7.249585588843074e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
