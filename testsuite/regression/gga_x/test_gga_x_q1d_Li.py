
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_q1d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q1d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.739167484982684e+00, -1.129906558229384e+00, -7.953235367639281e-03, -1.582188126356061e-01, -6.809903541078323e-03, -2.747492035203748e-07, -4.191278198031468e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_q1d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q1d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.456037253684431e+00, -2.457557097359117e+00, -2.119856689291547e+00, -2.120458689056647e+00, -4.780312056547542e-02, -4.741973143545082e-02, -2.116485321565448e-01, -9.634519517219565e-07, -4.475118841467034e-02, -7.763806265134299e-11, -1.129528606961323e-06, -1.067800213020925e-06, -1.973994144250996e-11, -8.485297925386701e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_q1d_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q1d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.369153859023728e-04, 0.000000000000000e+00, 2.350947305380267e-04, 3.235468522033646e-03, 0.000000000000000e+00, 3.220055032353621e-03, 1.802550424782952e-02, 0.000000000000000e+00, 1.784566145640225e-02, 3.016457438233386e-01, 0.000000000000000e+00, 6.169039325708831e-03, 7.805714334787368e+01, 0.000000000000000e+00, 3.936306851343091e-02, 6.270861060878678e-03, 0.000000000000000e+00, 5.855098279298559e-03, 2.865475171758597e-02, 0.000000000000000e+00, 4.101628817091765e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
