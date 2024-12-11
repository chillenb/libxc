
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_ft97_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_ft97", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.194305041821948e-02, -2.953893914300077e-04, -8.804857125179791e-06, -8.804948945328980e-06, -8.804954623664334e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_ft97_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_ft97", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.655891089137847e-02, 3.145067318711612e+06, -1.271755657725238e-02, 2.201429248136724e+06, -8.804997227993624e-06, 4.587340041527487e+05, -8.804920824165535e-06, 8.744863006198906e+03, -8.804954523969984e-06, 9.143728441156449e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_ft97_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_ft97", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.306849155175270e-02, 0.000000000000000e+00, -7.314555232301018e+30, 5.753976953684819e-03, 0.000000000000000e+00, -5.271113713732641e+30, 1.138387515390817e-14, 0.000000000000000e+00, -1.055224169379619e+30, 0.000000000000000e+00, 0.000000000000000e+00, -2.009847368039382e+28, 0.000000000000000e+00, 0.000000000000000e+00, -2.101439574963705e+24])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
