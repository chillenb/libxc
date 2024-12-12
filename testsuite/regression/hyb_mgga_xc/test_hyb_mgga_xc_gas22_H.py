
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_gas22_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_gas22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.555607122838748e-01, -3.305832177615029e-01, -1.716797333506530e-01, -5.199637417622219e-02, -8.840903036052129e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_gas22_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_gas22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.443038448815926e-01, -4.746517246134665e-01, -4.166412841611592e-01, -3.175650415208596e-01, -1.978693703913252e-01, -1.776644438220532e-01, -2.291058680965552e-02, 4.606476257381865e-02, -1.125783224345173e-02, 3.134013936147018e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_gas22_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_gas22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.097184947575508e-01, 0.000000000000000e+00, -2.353408570647588e+19, -9.324498151402433e-03, 0.000000000000000e+00, 8.486726623640647e+18, -4.427588660880569e-02, 0.000000000000000e+00, 2.378738780426329e+18, -1.013710971548977e+00, 0.000000000000000e+00, 1.006974308270862e+18, -7.995412413991638e-01, 0.000000000000000e+00, 6.789653736170149e+14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_gas22_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_gas22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-3.130014796507050e-01, -4.893634684325373e+02, -9.414836105082315e-02, 8.523965850560767e+04, -5.870589736776573e-02, -1.240399155929284e+05, -4.639095411928766e-02, -5.265226138146622e+05, -1.407038796222639e-05, -1.273480137488174e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
