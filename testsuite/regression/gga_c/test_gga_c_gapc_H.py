
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_gapc_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gapc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-3.208886229449724e-02, -1.964172850832493e-02, -1.122310648798043e-02, -1.195391332872265e-03, -4.835947154618815e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_gapc_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gapc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.693618913704207e-02, -2.237163216424908e-01, -3.889160390588324e-02, -1.852111972137594e-01, -2.723323159084904e-02, -1.212329846771509e-01, -4.756631260550569e-03, -1.686813759095082e-02, -2.312796134301424e-08, -3.870580763488558e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_gapc_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_gapc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.540749361302485e-02, 3.081498722604970e-02, 1.540749361302485e-02, 8.504456452048427e-03, 1.700891290409685e-02, 8.504456452048427e-03, 3.397321553040435e-02, 6.794643106080869e-02, 3.397321553040435e-02, 3.886288960525350e-01, 7.772577921050700e-01, 3.886288960525350e-01, 1.231501775652444e-02, 2.463003551304889e-02, 1.231501775652444e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
