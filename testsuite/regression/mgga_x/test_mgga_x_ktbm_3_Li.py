
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_3_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.913574201915202e+00, -1.277706722997498e+00, -2.575670890353670e-01, -1.755646651013831e-01, -5.557824250271421e-02, -1.165784556178047e-02, -2.133419546094143e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_3_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.674353478403008e+00, -2.676903662443195e+00, -1.862181749739251e+00, -1.863787321301494e+00, -3.373587492476893e-01, -3.373641147759328e-01, -2.411119021594982e-01, -1.340224255868104e-02, -7.727309323328868e-02, -4.250218699878443e-04, -1.409221766203943e-02, -1.399093928747369e-02, -2.838901334303650e-04, -2.015333523465336e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_3_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-5.068617728547783e-04, 0.000000000000000e+00, -5.051085650217837e-04, -1.969755421684841e-03, 0.000000000000000e+00, -1.964139259677309e-03, -5.898382341950657e-02, 0.000000000000000e+00, -6.116909834454862e-02, -7.829918575647364e+00, 0.000000000000000e+00, -3.674914230732291e+01, -7.684194753829313e+01, 0.000000000000000e+00, -9.217667746570236e+04, -6.843542923991061e-01, 0.000000000000000e+00, -3.285513983100135e+01, -1.395823410684971e+00, 0.000000000000000e+00, -2.119113846770193e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_3_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_3", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.770332970324055e-02, 1.768595793177849e-02, 2.331128865863001e-02, 2.330609425123722e-02, -1.125614971501305e-03, -1.104123004445108e-03, 1.978012835389190e-01, 4.696602901281611e-04, 2.653190658024318e-02, 3.755610614691539e-05, 1.016361797969982e-05, 4.777043089271041e-04, 1.694753710811931e-10, -6.660751522133844e-12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
