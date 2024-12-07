
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_w20_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_w20", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.244170333557998e-02, -3.115924478055232e-02, -2.526744856086759e-02, -1.362253242692258e-02, -1.362582918214600e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_w20_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_w20", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.640263148855331e-02, -2.229495265842297e-01, -3.503509370035879e-02, -2.133148421550090e-01, -2.871421788026111e-02, -1.691589340498080e-01, -1.607603254000085e-02, -8.443486563379121e-02, -1.699228193701940e-03, -6.969813299668394e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
