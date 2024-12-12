
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m06_2x_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.964716860175085e-02, -5.221548692514910e-02, -1.252781979125983e-03, 9.319270603164237e-04, 6.202430929280547e-08, 7.322397536706084e-03, -1.222838095435381e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m06_2x_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.228505605323582e-01, -1.222387938560969e-01, -1.278776259944709e-01, -1.274483748716757e-01, -1.138929264977927e-02, -1.275221486267673e-02, 1.644573139032035e-02, 5.297038382602518e-01, 2.356915469038894e-02, 3.356106684138701e-01, -9.858932136902992e-04, 3.770929376056029e-03, -5.148642523143301e-04, 4.466791793392861e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_2x_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.158196222449371e-05, 0.000000000000000e+00, -4.215794731563220e-05, 2.926020432158871e-04, 0.000000000000000e+00, 2.899756681545914e-04, -3.973254912023976e-02, 0.000000000000000e+00, -4.331036387375686e-02, -1.927364593221304e+01, 0.000000000000000e+00, 5.772735952423394e+02, -1.375327654331346e+02, 0.000000000000000e+00, 1.631757173838002e+06, 9.261431403586654e+00, 0.000000000000000e+00, 3.497727462950019e+02, 2.619858316059095e+01, 0.000000000000000e+00, 7.572569074439872e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m06_2x_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m06_2x", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([8.493370233514106e-03, 8.507266280266407e-03, 2.143734888218557e-03, 2.156998720820020e-03, 1.700307556746928e-02, 1.845265850682152e-02, 6.847462962578957e-01, -5.605378851728236e-03, 3.289063795642181e-01, -6.581649288187317e-04, -2.235697879287710e-06, -5.049643796865191e-03, -2.396890513612795e-14, -3.303371404008273e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
