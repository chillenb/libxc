
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mcml_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.872822706641400e+00, -1.340381763242511e+00, -3.990012356859167e-01, -1.665369886582610e-01, -7.987883129043548e-02, -9.180541722351176e-03, -1.313484423311580e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mcml_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.317488493032238e+00, -2.319504580105652e+00, -1.606964613226028e+00, -1.607552664029965e+00, -3.761463200198222e-02, -4.694241282984792e-01, -2.115596748376916e-01, -1.461282657547056e-02, -8.063660823892752e-02, 5.450937602966838e+01, -9.421823156400352e-03, -1.525697145754277e-02, -1.865453789663485e-04, 4.919796528493407e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mcml_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.056430277446415e-04, 0.000000000000000e+00, -3.052881380790661e-04, -9.389840677173994e-04, 0.000000000000000e+00, -9.471094369335167e-04, -6.228057957284713e-01, 0.000000000000000e+00, -3.184898883703481e-02, -4.788756356409086e+00, 0.000000000000000e+00, 2.109212165813075e-01, -5.660580756578908e+01, 0.000000000000000e+00, -1.105475567722963e+11, 8.992223191748917e-01, 0.000000000000000e+00, 2.020583723405953e-01, 4.101915029983998e+00, 0.000000000000000e+00, -9.512568718805497e+11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mcml_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mcml_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mcml", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.858654706967131e-05, -2.771836368078306e-11, -1.858127033203765e-04, -3.710932730947593e-17, 1.440200754513298e-01, 1.471229071769836e-10, -2.039937803724763e-02, 8.308842113909050e-12, -2.039668434101246e-06, 4.504098961671841e+01, 8.180389981854787e-16, 2.620517063602107e-12, 1.449772140275099e-33, 4.149686550878689e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
