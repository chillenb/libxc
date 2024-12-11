
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbe_r_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_r", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.795263721003819e+00, -1.285902659099054e+00, -4.463175734891911e-01, -1.600537476661825e-01, -8.322272994772741e-02, -2.555246611931571e-02, -4.776953343307693e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbe_r_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_r", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.239272891658953e+00, -2.241415083562093e+00, -1.510598806407568e+00, -1.511978173398983e+00, -3.587971781871697e-01, -3.589804913563068e-01, -2.052028731833261e-01, -3.244997387612120e-02, -7.014023148143540e-02, -1.032451199594545e-03, -3.411084356209616e-02, -3.386850326967900e-02, -6.896215677077075e-04, -4.902585286762272e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbe_r_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_r", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.628710476481774e-04, 0.000000000000000e+00, -2.619518861072791e-04, -1.070878448388437e-03, 0.000000000000000e+00, -1.067374662832542e-03, -1.145687322405982e-01, 0.000000000000000e+00, -1.143813431037365e-01, -4.020287792664901e+00, 0.000000000000000e+00, -6.650234491753813e-01, -8.933116327012895e+01, 0.000000000000000e+00, -4.259697353763270e+00, -6.756999564615129e-01, 0.000000000000000e+00, -6.310302208657527e-01, -3.100909283738345e+00, 0.000000000000000e+00, -4.438634919817114e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
