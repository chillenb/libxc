
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_ms2h_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_ms2h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.720663814735838e+00, -1.132780542999182e+00, -2.562897277805315e-01, -1.576315588660264e-01, -5.538187265809404e-02, -1.559200833401916e-02, -2.702519302947426e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_ms2h_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_ms2h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.414798269564066e+00, -2.417020052369047e+00, -1.704021058148058e+00, -1.705949401440084e+00, -3.430816915165407e-01, -3.430247139373239e-01, -2.146045985061856e-01, -1.981857015842723e-02, -7.496777747701426e-02, -6.294256156915860e-04, -2.086889665634192e-02, -2.068722971954683e-02, -4.204216423243509e-04, -1.930998165905293e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_ms2h_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_ms2h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.318944589051723e-04, 0.000000000000000e+00, -4.301243066830979e-04, -2.178963568751743e-03, 0.000000000000000e+00, -2.175817126152797e-03, -2.020901723404374e-01, 0.000000000000000e+00, -2.024593894276643e-01, -4.582388119459824e+00, 0.000000000000000e+00, -1.763881940737287e-01, -9.662787691281426e+01, 0.000000000000000e+00, -1.129514878190874e+00, -7.523647246385169e-05, 0.000000000000000e+00, -1.673748022840308e-01, 1.908556949686497e-08, 0.000000000000000e+00, -1.971036758696685e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_ms2h_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_ms2h", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.734700499886637e-02, 1.732072768530613e-02, 2.843194117191562e-02, 2.849179699459898e-02, 2.366031025931863e-04, 2.542672003290927e-04, 1.242061671668271e-01, 8.878958975883119e-18, 9.421834963427633e-03, 1.288715027945205e-18, -1.451714892817500e-20, 3.163746705027331e-18, -2.323554192327558e-18, 3.083943090314787e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
