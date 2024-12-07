
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_am05_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_am05", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-8.798912668290002e-02, -7.616414477544979e-02, -4.087521535286823e-02, -3.144406844686025e-02, -1.613848463932209e-02, -5.489733800067141e-03, -1.449162010750115e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_am05_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_am05", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.066579875448339e-01, -9.472187603559773e-02, -4.845345834675235e-02, -4.041034977954348e-02, -2.042891278905751e-02, -6.940105079828842e-03, -1.915940054487604e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_am05_Li_restr_1_vsigma():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_am05", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [8.662758621073029e-06, 2.784441330644804e-05, 4.246550987127993e-04, 1.692991136352967e+00, 1.881126704219453e+00, 1.873629830307053e-03, 1.442243994377536e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
