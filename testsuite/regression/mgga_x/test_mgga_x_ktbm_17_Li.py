
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_17_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.992133631954269e+00, -1.306151343968379e+00, -2.580856777295974e-01, -1.841758612536754e-01, -5.495265718909636e-02, -1.067548528040690e-02, -1.995554045844567e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_17_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.848337278928107e+00, -2.851106626765580e+00, -1.974399738594100e+00, -1.976183159489376e+00, -3.312046039265744e-01, -3.309823740746331e-01, -2.567087600732602e-01, -1.305863641713966e-02, -7.491067279058061e-02, -4.141153424365291e-04, -1.373060250459091e-02, -1.363226021942187e-02, -2.766051844064642e-04, -2.057056650570502e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_17_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-7.442803678636512e-04, 0.000000000000000e+00, -7.417820896606628e-04, -2.707993327410869e-03, 0.000000000000000e+00, -2.701142997523585e-03, -2.835012046721388e-02, 0.000000000000000e+00, -3.013058979354293e-02, -1.178774971390811e+01, 0.000000000000000e+00, -1.354733953516624e+01, -6.256439623244141e+01, 0.000000000000000e+00, -3.387922091133558e+04, -2.515473344496229e-01, 0.000000000000000e+00, -1.211406602284544e+01, -5.130261901066665e-01, 0.000000000000000e+00, 2.007983994225424e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_17_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_17", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.758536361882932e-02, 2.755867931281333e-02, 3.418531850928939e-02, 3.418833159176189e-02, -2.381565078696156e-03, -2.409215263854818e-03, 3.147576123616301e-01, 1.734171221713643e-04, 1.373544340235633e-02, 1.380367470234503e-05, 3.735967986532028e-06, 1.764371613551340e-04, 6.228961577676401e-11, -9.085814937075001e-12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
