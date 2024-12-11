
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_wb97x_d3_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x_d3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.396747679249838e+00, -9.701536043717317e-01, -2.567009839678090e-01, -5.826697485496091e-02, -1.119758229185084e-02, 8.145826695534287e-03, 1.265438915304352e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_wb97x_d3_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x_d3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.813979007517820e+00, -1.815608831740609e+00, -1.193665630740200e+00, -1.194655691045354e+00, -3.006478506304472e-02, -2.913439822049956e-02, -1.046399666279473e-01, 3.777252345457995e-01, -8.617596894535466e-03, 2.314265471100583e-01, 9.926715414995365e-03, 1.033386290701463e-02, 4.625820028673840e-05, 5.113612269840187e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_wb97x_d3_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x_d3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.170126373550894e-04, 0.000000000000000e+00, -1.164395525372641e-04, -6.532261536409937e-04, 0.000000000000000e+00, -6.508793900666472e-04, -1.740150714825920e-01, 0.000000000000000e+00, -1.745012126722522e-01, 6.003743611874404e+00, 0.000000000000000e+00, 6.394920117848224e+01, -2.133950380535919e+01, 0.000000000000000e+00, 7.567463029442082e+03, 7.144528729966920e-01, 0.000000000000000e+00, 7.661716164834018e-01, 1.188333495242950e+00, 0.000000000000000e+00, 1.985443723860460e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
