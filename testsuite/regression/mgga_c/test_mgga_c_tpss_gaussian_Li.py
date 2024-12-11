
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_tpss_gaussian_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.346210030555922e-02, -8.371482262002407e-02, -4.959806172627841e-02, -1.808612568676710e-02, -1.095911360423736e-02, -5.892287584479840e-12, -8.300292557722951e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_tpss_gaussian_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.026712126809282e-01, -1.025093304209565e-01, -9.254539378426754e-02, -9.240668603323736e-02, -5.664537600732751e-02, -5.668890672673786e-02, -2.101631195919127e-02, -1.243108863726644e-01, -1.310473963822716e-02, -7.152742107203572e-02, -4.266311410392269e-11, -3.375223960418613e-11, -1.060211934891276e-18, -4.836050979090974e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpss_gaussian_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.386289308112864e-04, 2.772579476777345e-04, 1.386289306877445e-04, 5.969305200499265e-04, 1.193861040099853e-03, 5.969305200499265e-04, 1.796836956126973e-01, 3.593673912253946e-01, 1.796836956126973e-01, 4.170979357197962e+00, 8.341958721679674e+00, 4.170980071719414e+00, 1.681725806323406e+02, 3.363451612646812e+02, 1.681725806323406e+02, 4.971679826635226e-09, 1.079283234086868e-08, 5.396416269876317e-09, -6.849954942182521e-16, -8.504358669295532e-15, -4.346578043363977e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpss_gaussian_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss_gaussian", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.180368202439456e-09, -1.180368202439457e-09, -9.113724352682790e-88, -9.113724352682788e-88, -2.843802031518637e-80, -2.843802031518635e-80, -3.797175796549215e-10, -3.797175796548377e-10, -2.940009116187429e-25, -2.940009113812175e-25, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
