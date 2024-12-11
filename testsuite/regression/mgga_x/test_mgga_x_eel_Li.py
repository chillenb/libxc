
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_eel_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_eel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.037997707450265e+00, -1.367480991417158e+00, -7.972543138921706e-02, -1.841997647524651e-01, -2.774209861354482e-02, -4.676583198523309e-03, -1.247995907010869e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_eel_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_eel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.727621836912313e+00, -2.729607472371634e+00, -2.078694731810219e+00, -2.079721373933420e+00, -1.395895531703742e-01, -1.435427468943682e-01, -2.456787038182913e-01, 4.473958317173745e-02, -7.883613702262554e-02, 5.480070383076755e-04, 2.807335340454680e-02, 4.690336253942507e-02, 1.768429384059110e-05, -5.690629744801713e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_eel_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_eel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.486450744221131e-07, 0.000000000000000e+00, 2.372810501702936e-07, 6.993501788565315e-05, 0.000000000000000e+00, 6.918059294075585e-05, 9.216927196429980e-01, 0.000000000000000e+00, 9.204117240858739e-01, 0.000000000000000e+00, 0.000000000000000e+00, -1.399497661154355e+03, 3.121977013318648e+02, 0.000000000000000e+00, -1.302129261252010e+06, -1.504896367054401e+01, 0.000000000000000e+00, -1.257668253012355e+03, -9.441035401344461e-01, 0.000000000000000e+00, 9.210157177418379e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_eel_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_eel", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.639117282964233e-03, 1.571326688883582e-03, 3.748617966059754e-02, 3.736444473746692e-02, 6.105723574708316e-03, 6.878537439489374e-03, 0.000000000000000e+00, 1.822751267225574e-02, 3.503808299746768e-01, 5.410275944576292e-04, 2.279155202050461e-04, 1.863471697208998e-02, 1.168967088668256e-10, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
