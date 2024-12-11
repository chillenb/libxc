
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_bc95_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_bc95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.999257936936117e-02, -6.482501445426840e-02, -2.027784815737483e-01, -2.934998413999233e-03, -3.853961990161335e-02, -1.313477194378342e-07, -1.895103513987220e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_bc95_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_bc95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.572884380794766e-02, -4.549008938781305e-02, -3.520101165826842e-02, -3.507759391776805e-02, 6.444257119921555e-02, 6.212132439592535e-02, 1.476271034065496e-03, -8.035062409827978e-04, 1.814758441981495e-02, -1.198433345089606e-06, -8.526226735348615e-07, -5.797613237153149e-07, 5.153166634824347e-14, 2.492224891642061e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_bc95_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_bc95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.093130027654306e-04, 0.000000000000000e+00, 1.088952905248786e-04, 8.373428660375165e-04, 0.000000000000000e+00, 8.326756155872380e-04, 4.959454231912993e+00, 0.000000000000000e+00, 4.856401913746009e+00, 5.260941534372241e+00, 0.000000000000000e+00, 5.649063800257172e+00, 2.416488231143508e+03, 0.000000000000000e+00, 6.632508543915026e+02, 1.193814816847380e-04, 0.000000000000000e+00, 8.771345115992075e-03, 5.497290808982469e-11, 0.000000000000000e+00, 3.225550206069089e+20])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_bc95_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_bc95", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-2.873107274983154e-03, -2.871683944547766e-03, -5.332155022208856e-03, -5.324538201822170e-03, -5.428445283005290e-02, -5.432066785617735e-02, -1.392078133987678e-01, -1.176431803685671e-07, -5.378184861852190e-01, -3.855053355621430e-11, -5.743566782136254e-11, -1.256638198788993e-07, -5.361832595959912e-22, -6.047408678635331e+02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
