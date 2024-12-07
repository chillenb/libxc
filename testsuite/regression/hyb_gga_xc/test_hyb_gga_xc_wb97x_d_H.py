
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_wb97x_d_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.274903125183795e-01, -3.769718088762509e-01, -2.002182760359870e-01, -5.435405500464274e-02, -1.311817146684817e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_wb97x_d_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.038040502725850e-01, -2.607781774937694e-01, -5.278334667664825e-01, -2.604026678384159e-01, -2.402175794506674e-01, -2.095617293812584e-01, 9.827234510421732e-03, 2.113313792951359e-02, -1.692194890867323e-03, 2.583380831405693e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_wb97x_d_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_wb97x_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.109763053132451e-01, 0.000000000000000e+00, -2.877543789948519e+20, 8.757755819203690e-04, 0.000000000000000e+00, -1.569061823272969e+20, -1.090212220697302e-01, 0.000000000000000e+00, 8.450050337535730e+18, -1.132912086700970e+01, 0.000000000000000e+00, 4.718007309670403e+19, -7.021590660797574e-01, 0.000000000000000e+00, 8.925456674450025e+13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
