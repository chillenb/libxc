
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbe_mol_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_mol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.808557208788272e+00, -1.301675968127581e+00, -4.290424040396070e-01, -1.608003857945516e-01, -8.322432598882758e-02, -2.054701298665067e-02, -3.838587364874669e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbe_mol_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_mol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.227255292673350e+00, -2.229394434533376e+00, -1.503575176561923e+00, -1.504936682005551e+00, -4.230761225852329e-01, -4.232957544970982e-01, -2.043905971784653e-01, -2.612457953552514e-02, -7.847216577635698e-02, -8.296440358997409e-04, -2.746760587137359e-02, -2.726974682921844e-02, -5.541557105367900e-04, -3.939542797631680e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbe_mol_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_mol", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.138469688585424e-04, 0.000000000000000e+00, -3.127775341660386e-04, -1.218675898638558e-03, 0.000000000000000e+00, -1.214818783564852e-03, -7.225480499171139e-02, 0.000000000000000e+00, -7.204563981284504e-02, -4.901119921877358e+00, 0.000000000000000e+00, -2.211284179093445e-01, -7.110360792832866e+01, 0.000000000000000e+00, -1.413763784681271e+00, -2.247266474291468e-01, 0.000000000000000e+00, -2.098494016753059e-01, -1.029167091096737e+00, 0.000000000000000e+00, -1.473146519968143e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
