
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_xc_otpss_d_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.695453076912437e-01, -6.110276197428147e-01, -3.681939030732592e-01, -1.347726345451777e-01, -7.291287554335777e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_xc_otpss_d_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-9.025118419041500e-01, -6.386549589809801e-02, -7.874506521106464e-01, -3.151546687136125e-02, -4.592868384947269e-01, -1.227438831400666e-02, -1.283115311462504e-01, -1.865956298347723e-04, -9.718280456862727e-03, -2.644778884281322e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_otpss_d_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.157856175593073e-01, 5.224551932316054e+00, 2.612275966158029e+00, -1.286832055183138e-02, 7.807489878336960e-02, 3.903744939168480e-02, -4.904139100701059e-02, 1.429709853062718e-01, 7.148549265313589e-02, -5.645965848081706e+00, 1.120170456113466e-01, -2.756037418640079e+00, -2.745506310125044e+00, 1.127149641496277e-03, 4.707116728966883e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_xc_otpss_d_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_xc_otpss_d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.013904920318700e+00, -6.157197670080524e+00, 5.767201395905502e-04, -6.602259235694637e-02, -1.103877976508695e-02, -2.450315541283260e-02, -1.446374620185123e-03, -3.669793689190921e-04, -4.853154689347611e-10, -3.861036586606974e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
