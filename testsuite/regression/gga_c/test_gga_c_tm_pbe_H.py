
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_tm_pbe_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tm_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.278977111878125e-02, -2.163962778489024e-02, -8.099759524103915e-03, -1.116981777017495e-04, -3.223118256664992e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_tm_pbe_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tm_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.603981338273911e-02, -1.748956923254752e+00, -6.048340011612403e-02, -9.001139824839546e+00, -3.600148280253303e-02, 2.121500726060714e+01, -7.010634702433981e-04, 9.804835754409362e-02, -2.084949557836294e-09, 1.094679965814652e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_tm_pbe_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_tm_pbe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.272969807807973e-02, -2.545939615615946e-02, -1.272969807807973e-02, 1.856435906781372e-02, 3.712871813562743e-02, 1.856435906781372e-02, 6.319626560257455e-02, 1.263925312051493e-01, 6.319626560257455e-02, 6.830214319079723e-02, 1.366042863815998e-01, 6.830214319079723e-02, 1.373924130987176e-03, 2.747848260409458e-03, 1.373924130987176e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
