
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_op_pw91_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.282491597618938e-05, -4.282921842479352e-05, -4.286806023971772e-05, -4.303749812830167e-05, -1.903057095108643e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_op_pw91_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.280509733622089e-05, -4.568378406098233e+04, -4.281388317783192e-05, -3.293883518866821e+04, -4.285533235349315e-05, -6.602234304951186e+03, -4.303548413896488e-05, -1.266919358788824e+02, 9.816327384436113e-03, -1.631627467575074e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_op_pw91_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_op_pw91", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.439444435294348e-09, 0.000000000000000e+00, 5.477205581615586e+19, 2.790902123151837e-09, 0.000000000000000e+00, 3.948610001738568e+19, 4.712334933813306e-08, 0.000000000000000e+00, 7.904548557588153e+18, 1.000553530677438e-05, 0.000000000000000e+00, 1.508507636230411e+17, -8.351360892553330e+03, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
