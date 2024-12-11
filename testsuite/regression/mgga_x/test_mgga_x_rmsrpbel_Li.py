
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_rmsrpbel_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmsrpbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.836230605431333e+00, -1.232404712179120e+00, -2.693988052111181e-01, -1.674504351513954e-01, -5.876511978853868e-02, -2.055687026651119e-02, -3.458853198660017e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_rmsrpbel_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmsrpbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.532591582920766e+00, -2.534934893134806e+00, -1.774755734320094e+00, -1.776551562736194e+00, -3.631081941000330e-01, -3.632220358081044e-01, -2.264276368030667e-01, -2.615912580860029e-02, -8.179743012021383e-02, -8.296468243504021e-04, -2.750809938117428e-02, -2.730803076839329e-02, -5.541564195188991e-04, -2.024011734688082e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rmsrpbel_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmsrpbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.799931453445607e-04, 0.000000000000000e+00, -3.784757060148854e-04, -1.768875724710445e-03, 0.000000000000000e+00, -1.765678119963678e-03, -2.222459689300139e-01, 0.000000000000000e+00, -2.227952743597900e-01, -4.434429135410118e+00, 0.000000000000000e+00, -3.092674806204462e-192, -1.132055065961170e+02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, -6.754522588623127e-181, 0.000000000000000e+00, 0.000000000000000e+00, -2.148837620409553e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_rmsrpbel_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_rmsrpbel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.218644372094369e-02, 1.216763660387726e-02, 1.935302338819563e-02, 1.938790617083574e-02, 7.034216443653433e-04, 7.566541676296207e-04, 8.804556179668979e-02, 0.000000000000000e+00, 2.883586719360815e-02, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 8.628323449201401e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
