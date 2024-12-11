
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_ccalda_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_ccalda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.247092647740938e-02, -3.118079983566957e-02, -2.518518753711948e-02, -1.327196219152886e-02, -1.569790621002125e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_ccalda_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_ccalda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.561496192152709e+00, 2.335235458133509e+00, -3.508811108098497e-02, -2.506001669966031e-01, -2.872211520383755e-02, -1.951180619425772e-01, -1.571598845292717e-02, -9.051731108365549e-02, -2.001663505771958e-03, -7.071279902178262e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_ccalda_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_ccalda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.062641756855889e+02, -2.125283513711778e+02, -1.062641756855889e+02, 3.861716962089102e-02, 7.723433924178204e-02, 3.861716962089102e-02, 1.466783821777844e-01, 2.933567643555688e-01, 1.466783821777844e-01, 3.984089201582504e+00, 7.968178403165006e+00, 3.984089201582504e+00, 3.345362348910659e+03, 6.690724697821317e+03, 3.345362348910659e+03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_ccalda_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_ccalda", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.534384229912539e+02, 2.533107088675942e+02, -1.730819472220531e-17, -1.248422449993562e-17, 2.892449385473158e-18, -7.932162685269107e-18, -3.247808060672744e-18, -4.268771574740442e-18, -4.025191464041414e-19, -2.720796370971170e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
