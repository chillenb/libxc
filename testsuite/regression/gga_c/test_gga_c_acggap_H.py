
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_acggap_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_acggap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.208213996193481e-02, -1.907107935379267e-02, -9.389200751568041e-03, -3.542824732291998e-04, -1.262175851682312e-09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_acggap_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_acggap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.694215615916235e-02, 8.059120922087109e+00, -4.073355345908318e-02, 3.282327730543197e+02, -2.886242951291023e-02, 1.974379923543794e+02, -1.926387652722024e-03, 3.501245508808879e+00, -7.983449317643457e-09, 3.738376420597909e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_acggap_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_acggap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.561964892873798e-02, 3.123929785747596e-02, 1.561964892873798e-02, 9.586311553493397e-03, 1.917262310698679e-02, 9.586311553493397e-03, 4.221430334274946e-02, 8.442860668549892e-02, 4.221430334274946e-02, 1.795266771990623e-01, 3.590533543981246e-01, 1.795266771990623e-01, 5.183978820878302e-03, 1.036795764175660e-02, 5.183978820878302e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
