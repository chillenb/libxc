
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_rge2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_rge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.214659045493441e-02, -2.000767267723822e-02, -9.793986907312535e-03, -2.418325762801581e-04, -3.692302223516630e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_rge2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_rge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.687172107132694e-02, 1.210927329348189e+00, -4.159896637539252e-02, 6.307985465005881e+01, -3.030984542009359e-02, 4.095663030751704e+01, -1.412340668184891e-03, 5.020924101579524e-01, -2.402172844320760e-09, 1.690874230106214e-08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_rge2_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_rge2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.312871018707105e-02, 2.625742037414209e-02, 1.312871018707105e-02, 9.440971195696148e-03, 1.888194239139229e-02, 9.440971195696148e-03, 4.395170292060625e-02, 8.790340584121249e-02, 4.395170292060625e-02, 1.314886856847675e-01, 2.629773713695363e-01, 1.314886856847675e-01, 1.573193973827253e-03, 3.146387949116616e-03, 1.573193973827253e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
