
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_23_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_23", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.183805400679274e+00, -1.540645161655021e+00, -2.724467638307014e-01, -1.945284325814549e-01, -6.583832172449899e-02, -9.552643395963376e-03, -1.792069686132928e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_23_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_23", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.621343759736568e+00, -2.623866012680602e+00, -1.773559557156544e+00, -1.774762891007552e+00, -3.710195329248297e-01, -3.736652535365166e-01, -2.429639461781834e-01, -1.178263229489187e-02, -9.174880454963215e-02, -3.736015968154110e-04, -1.287469464741470e-02, -1.230030534707306e-02, -2.594526694464513e-04, -1.774031918745074e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_23_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_23", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.893215172811433e-04, 0.000000000000000e+00, -8.863166324217154e-04, -3.466727546122768e-03, 0.000000000000000e+00, -3.457735755726782e-03, -5.565410800012979e-02, 0.000000000000000e+00, -5.896698213815700e-02, -1.359176727123858e+01, 0.000000000000000e+00, -7.633648903577449e+00, -9.394701623779281e+01, 0.000000000000000e+00, -1.902429343007722e+04, 2.718236161717663e-01, 0.000000000000000e+00, -6.827480347606440e+00, 5.749002405942439e-01, 0.000000000000000e+00, -8.612506718877748e+04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_23_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_23", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([3.268667414231245e-02, 3.264407311269218e-02, 5.532721210965078e-02, 5.528412399510541e-02, 2.248452365672979e-02, 2.438688804304065e-02, 3.255633245928075e-01, 9.805023993908597e-05, 3.926686724992028e-01, 7.751283718390969e-06, -8.266215601509986e-08, 9.979964321730089e-05, -5.527188508836253e-16, 3.757062498525063e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
