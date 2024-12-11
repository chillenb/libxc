
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_gx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.050098287120520e+00, -1.356759460930664e+00, -2.567524314635121e-01, -1.884554087194336e-01, -5.708775066766753e-02, -1.404615694964276e-02, -2.458047312553872e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_gx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.884420667347992e+00, -2.887114890579095e+00, -2.036799974557977e+00, -2.038299680965265e+00, -3.571022557482202e-01, -3.575197128143131e-01, -2.596682341478578e-01, 1.026153714786147e+01, -8.140822723397724e-02, 1.283094151621529e+02, 4.741240024072703e+02, 1.008882393225258e+01, 2.843369553030764e+07, -1.860583909431204e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.181509457288300e-04, 0.000000000000000e+00, -4.166315057145197e-04, -1.925964283633614e-03, 0.000000000000000e+00, -1.918415174523661e-03, -1.119630748910025e-02, 0.000000000000000e+00, -1.170696370160897e-02, -6.163645769270255e+00, 0.000000000000000e+00, -2.632781328878477e+05, -1.852544756675566e+01, 0.000000000000000e+00, -2.602164060191174e+11, -2.153105730340034e+05, 0.000000000000000e+00, -2.216900296823392e+05, -1.307311823193295e+12, 0.000000000000000e+00, -1.116396613712493e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gx_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.169740640016985e-02, 2.167705367572001e-02, 3.344650858479087e-02, 3.340055085459105e-02, 2.695692524975798e-03, 2.815046749766154e-03, 2.366053314583921e-01, 3.362509180130806e+00, 4.430316555580478e-02, 1.060213792477734e+02, 3.197614617308751e+00, 3.221041510522955e+00, 1.587286433637512e+02, 4.870078892194039e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
