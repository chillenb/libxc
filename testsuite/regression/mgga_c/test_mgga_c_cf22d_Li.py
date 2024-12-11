
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_cf22d_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.229711442247070e-01, -1.198060841469819e-01, -1.931229797705034e-01, -4.284362660428810e-02, -3.894013191924937e-02, -1.333774619301220e-02, -3.309387292989996e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_cf22d_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.965642105632781e-01, -3.961779940392341e-01, -3.393540810058333e-01, -3.391555733584219e-01, -1.364573074007791e-01, -1.366268056019483e-01, -7.985813487967680e-02, -3.245491342262845e-01, -2.082115985984613e-02, -2.284097483352809e-01, -1.676482825961988e-02, -1.695295459865665e-02, -3.893153579553492e-04, -5.712657444323047e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cf22d_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.245202046562156e-04, 8.490404093124313e-04, 4.245202046562156e-04, 6.419179523206745e-04, 1.283835904641349e-03, 6.419179523206745e-04, 7.980788181443537e-01, 1.596157636288707e+00, 7.980788181443537e-01, 1.253902025037972e+01, 2.507804050075944e+01, 1.253902025037972e+01, 7.567562883287485e+02, 1.513512576657497e+03, 7.567562883287485e+02, 2.945411577746500e-08, 5.890823314954959e-08, 2.945411577746500e-08, 3.102176096667578e-16, -5.487440338489425e-15, 3.102176096667578e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_cf22d_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_cf22d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.156077385232313e-02, 2.156077385232312e-02, 3.026046339861434e-02, 3.026046339861433e-02, -1.547168687704779e-02, -1.547168687704777e-02, 8.164737457301386e-01, 8.164737457299610e-01, -2.155435100643654e-01, -2.155435099157890e-01, -2.010399497785979e-08, -2.010399497731639e-08, -5.310701623239670e-20, -5.310378509244584e-20])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
