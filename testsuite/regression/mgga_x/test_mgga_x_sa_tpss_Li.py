
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_sa_tpss_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_sa_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.750850113171398e+00, -1.209542605279631e+00, -2.978605410770948e-01, -1.585790712681438e-01, -6.346118676546005e-02, -2.054985979597249e-02, -3.506166968767713e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_sa_tpss_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_sa_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.349510730174100e+00, -2.351643841676746e+00, -1.628184020331381e+00, -1.629604217433139e+00, -3.935648691745789e-01, -3.932108901880116e-01, -2.124431693841069e-01, 2.388887961171043e+01, -8.272498631881475e-02, 2.992527978744955e+02, 1.105746220519460e+03, 2.348414023466793e+01, 6.629777845738985e+07, -2.263484506481381e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_sa_tpss_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_sa_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-3.430579072349421e-04, 0.000000000000000e+00, -3.403982921869200e-04, -9.613563258861719e-04, 0.000000000000000e+00, -9.593463819445512e-04, -4.080568632117617e-02, 0.000000000000000e+00, -4.151271418711959e-02, -7.449814788452831e+00, 0.000000000000000e+00, -6.125154175034319e+05, -3.287703508174576e+01, 0.000000000000000e+00, -6.068951315392974e+11, -5.021372793461552e+05, 0.000000000000000e+00, -5.156818336986079e+05, -3.048209809237835e+12, 0.000000000000000e+00, -1.701817018929629e+11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_sa_tpss_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_sa_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.321097835798142e-03, 2.302274766820900e-03, 2.362976817476223e-03, 2.367170267371585e-03, -6.728064687841221e-04, -7.108091305058498e-04, 2.682093493597700e-02, 7.822859259119674e+00, -1.582443924128943e-02, 2.472705694799261e+02, 7.457327716558623e+00, 7.492586586635507e+00, 3.701016078371850e+02, -1.379374666938812e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
