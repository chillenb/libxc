
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_jk_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.739643652199130e+00, -1.280756889578476e+00, -4.217812076049586e-01, -1.589377621617678e-01, -7.984564191306534e-02, -5.816741491527952e-02, -6.030845632235828e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_jk_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.310486859758587e+00, -2.312570209628291e+00, -1.593513488805315e+00, -1.594854803637054e+00, -5.238837766381728e-01, -5.178896002238079e-01, -2.075039804942802e-01, -1.806880562720344e-01, -9.543813391332703e-02, -1.511431682499261e-02, -1.752620700740294e-02, -1.653726945684815e-01, -3.075605911911221e-04, -1.064272437764390e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_jk_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.339300468628764e-05, 0.000000000000000e+00, -2.333246214425397e-05, -8.149571393952599e-04, 0.000000000000000e+00, -8.127695325626180e-04, 5.979609576006020e-02, 0.000000000000000e+00, 5.397732846895865e-02, -3.373601091542566e+00, 0.000000000000000e+00, 2.657890176781309e+03, 2.464648820568092e+00, 0.000000000000000e+00, -6.042785480610636e+05, 8.698676718437183e+01, 0.000000000000000e+00, 1.965649205387484e+03, 3.668187438801037e+03, 0.000000000000000e+00, -4.514550412709048e+07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_jk_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_jk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([-2.074812781156013e-06, -2.074751591264397e-06, -1.029962917292421e-03, -1.029316303338758e-03, -6.003955212202656e-03, -5.798303609702818e-03, -2.094066303753966e-03, -6.726176756045944e-03, -2.897566222142534e-02, -2.526293169101186e-03, -4.645731282678658e-06, -6.063681510399129e-03, -6.076105760564746e-13, -2.082817774148401e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
