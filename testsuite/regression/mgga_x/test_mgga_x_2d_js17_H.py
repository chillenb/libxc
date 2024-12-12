
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_2d_js17_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.428328583775327e-01, -7.058234732322629e-01, -3.938798686151287e-01, -3.515993032340101e-01, -1.008491961774242e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_2d_js17_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.275274170536924e+00, 2.892157210629630e-16, -9.605897993498393e-01, -2.259872614254025e-16, -2.982905602821953e-01, -1.407818864669780e-16, 8.838619304012517e-02, -2.746437157075371e-17, 3.309106990764097e-01, 2.067878000928716e-17])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_2d_js17_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.484651863567872e-01, 0.000000000000000e+00, 0.000000000000000e+00, -4.254249261838458e-02, 0.000000000000000e+00, 0.000000000000000e+00, -5.840342128065633e-01, 0.000000000000000e+00, 0.000000000000000e+00, -6.347186496506744e+01, 0.000000000000000e+00, 0.000000000000000e+00, -1.350016734123281e+06, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_2d_js17_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_2d_js17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([6.627099319148250e-03, 0.000000000000000e+00, 5.210020101952359e-03, 0.000000000000000e+00, 8.336341004848881e-03, 0.000000000000000e+00, 1.828637387121933e-02, 0.000000000000000e+00, 4.160762769891738e-02, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
