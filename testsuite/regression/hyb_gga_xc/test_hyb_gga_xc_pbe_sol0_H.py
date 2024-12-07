
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_pbe_sol0_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_sol0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-4.985800880706420e-01, -4.477513123455309e-01, -2.703046131983495e-01, -9.396000359953065e-02, -5.545947030055777e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_pbe_sol0_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_sol0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-6.582080810966536e-01, 1.014698796252341e+00, -5.880832451534993e-01, 5.660683950431883e+01, -3.407617938996514e-01, 3.977148618814145e+01, -9.188756002564415e-02, 6.346862779076641e-01, -7.385370517980491e-03, 2.244372685477213e-08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_pbe_sol0_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_pbe_sol0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [4.313609707191189e-03, 2.285192576044089e-02, 1.142596288022045e-02, -1.550298967905775e-03, 1.780590821288795e-02, 8.902954106443973e-03, -3.609407409753492e-02, 8.851829840055765e-02, 4.425914920027881e-02, -3.747358119761019e+00, 3.337722534823843e-01, 1.668861267411923e-01, -7.371920397727334e+00, 4.176370345584319e-03, 2.088185172073023e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
