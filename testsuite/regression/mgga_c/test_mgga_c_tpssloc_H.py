
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_tpssloc_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpssloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.189080403164646e-15, -3.118079983567873e-02, -2.518518753715506e-02, -1.327196219237759e-02, -1.569791198298438e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_tpssloc_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpssloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.447452267070777e-02, -6.447452718751850e-02, -3.508811108098649e-02, -2.505967048220198e-01, -2.872211520384434e-02, -1.951133609024166e-01, -1.571598845314862e-02, -9.050877629030626e-02, -2.001663746557634e-03, -7.057704296362531e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpssloc_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpssloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.637213811050588e+00, 5.274427622101179e+00, 2.637213811050588e+00, 1.467457467323178e-02, 2.934914934646356e-02, 1.467457467323178e-02, 1.253865252944781e-01, 2.507730505889562e-01, 1.253865252944781e-01, 2.465088160899342e+01, 4.930176321798683e+01, 2.465088160899342e+01, 5.004622181659656e+06, 1.000924436331931e+07, 5.004622181659656e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpssloc_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpssloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-6.289714337416681e+00, -6.285196559897430e+00, -3.801689668808897e-74, -3.792431798187254e-74, -1.595100320873994e-71, -1.595011957941993e-71, -4.487811461307024e-57, -4.487797224325138e-57, -7.265458699613291e-42, -7.265458699119815e-42]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
