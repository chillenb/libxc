
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_regtm_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.941159319547109e+00, -1.359982512890910e+00, -3.810491008219813e-01, -1.748537368915256e-01, -7.696889665311719e-02, -7.596817714725745e-02, -3.283654011231219e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_regtm_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.512778968580343e+00, -2.515185362758397e+00, -1.700436603153351e+00, -1.702126540217410e+00, -3.548542747172230e-01, -3.550844190934406e-01, -2.297869600416375e-01, -2.085620783475745e-02, -7.835898599028496e-02, -2.079140640245217e-03, -1.100575863700110e-01, -2.155148277148765e-02, -4.368028490904265e-02, -1.323262530378716e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_regtm_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.273199174992000e-04, 0.000000000000000e+00, -2.264197017943430e-04, -9.459975398677656e-04, 0.000000000000000e+00, -9.409423976708569e-04, -1.187167354090842e-01, 0.000000000000000e+00, -1.193030530107305e-01, -3.430764561171702e+00, 0.000000000000000e+00, -5.047770576772504e+02, -8.231108469027701e+01, 0.000000000000000e+00, -4.240807153799863e+06, 7.188759654607427e+00, 0.000000000000000e+00, -4.456805425096433e+02, 8.451602987215786e+01, 0.000000000000000e+00, -2.576093862008622e+07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_regtm_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_regtm", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([8.301898104101854e-03, 8.294439420590924e-03, 9.864522248953251e-03, 9.829863839803854e-03, 1.681535682779378e-02, 1.720746298084494e-02, 1.035457731647049e-01, 3.895036744553969e-03, 1.117715736527770e-01, 1.039469549318500e-03, -1.501912709409667e-04, 3.912939835595237e-03, -4.876834669210909e-08, 6.760051167976740e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
