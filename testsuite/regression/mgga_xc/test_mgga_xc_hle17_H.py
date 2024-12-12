
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_hle17_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_hle17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-8.803442360020739e-01, -7.901520617203884e-01, -4.632859409625064e-01, -1.643138201574123e-01, -9.245768364875596e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_hle17_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_hle17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.068770766877459e+00, -3.206403344640970e-02, -9.869042212427648e-01, -1.817837156035678e-02, -5.873236664155995e-01, -8.087243827794363e-03, -1.568771555012921e-01, -1.586025621996512e-04, -1.231902433162916e-02, -2.331911483730140e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_hle17_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_hle17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.292275478925710e+00, 2.623015054173823e+00, 1.311507527086912e+00, -8.253439367478900e-02, 4.503387712629729e-02, 2.251693856314864e-02, -1.339556290798780e-01, 9.419907197781073e-02, 4.709953598890539e-02, -7.012253609561853e+00, 9.521225331871966e-02, 1.647694380861689e+00, -6.925324834280055e+00, 9.938099181325841e-04, 5.181010068625027e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_hle17_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_hle17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.023208671990282e+01, -3.091254980230632e+00, 1.419020685285897e-01, -3.808206425187470e-02, 3.720043583679215e-02, -1.614435611255827e-02, 1.062113504499647e-04, -3.119251400424826e-04, -2.834785078403128e-10, -3.404283064528696e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
