
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mbeefvdw_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.992858502667971e+00, -1.383768925758582e+00, -3.563410548056445e-01, -1.801421230704198e-01, -7.364443275224271e-02, -1.431923654349070e-02, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mbeefvdw_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.657739976486759e+00, -2.660158384689454e+00, -1.839951883089232e+00, -1.841647817314982e+00, -2.120003286770225e-01, -4.107227162496138e-01, -2.404334003867731e-01, -1.869510238514864e-02, -8.913347002815948e-02, -1.995787095311648e-17, -1.871218407795552e-02, -1.951886238528263e-02, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbeefvdw_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.029497001279894e-06, 0.000000000000000e+00, 3.106925760094476e-06, -2.359177045222050e-05, 0.000000000000000e+00, -2.184932393750618e-05, -3.006916711565872e-01, 0.000000000000000e+00, -3.215310319705828e-02, 4.062554317973009e-02, 0.000000000000000e+00, 2.692127310799807e-01, -1.982367790380524e+01, 0.000000000000000e+00, 0.000000000000000e+00, 5.675889975002828e-02, 0.000000000000000e+00, 2.552431466735469e-01, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbeefvdw_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbeefvdw", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([3.751185308869239e-06, 3.636077279658619e-12, 2.768494197372811e-05, 5.527908799659043e-18, 6.517821003031041e-02, 2.041037154969231e-10, 2.466826736429825e-03, 6.253848627225541e-12, 6.535522065204662e-07, 0.000000000000000e+00, -7.634232460839278e-16, 1.977389772425655e-12, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
