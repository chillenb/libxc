
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hcth_407p_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_407p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-6.963308327759332e-01, -6.076437612494550e-01, -3.651290375719724e-01, -1.911828193292820e-01, -1.521225855275990e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hcth_407p_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_407p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.258501577563938e-01, -1.899871755578935e-01, -8.206666969484370e-01, -2.025456703629675e-01, -4.377486292481509e-01, -1.817663484212870e-01, -8.897890061066797e-02, 1.995436585695850e-02, -2.011501545965729e-02, 1.060985445974538e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hcth_407p_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hcth_407p", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.129686025449172e-02, 0.000000000000000e+00, -4.976833886776428e+20, 6.079033115158842e-03, 0.000000000000000e+00, -3.659028472192109e+20, -9.992513917605672e-02, 0.000000000000000e+00, -1.678269754991724e+20, -1.836540971122109e+01, 0.000000000000000e+00, 1.120873172504149e+20, -2.542244910679832e+01, 0.000000000000000e+00, 4.622774346926970e+14]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
