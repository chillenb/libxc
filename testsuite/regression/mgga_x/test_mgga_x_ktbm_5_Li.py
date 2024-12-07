
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_5_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_5", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.965621774221892e+00, -1.384910456312683e+00, -3.528834798432905e-01, -1.763280578527952e-01, -7.592789063871193e-02, -1.328131672001611e-02, -2.425010427169956e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_5_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_5", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.465039300492368e+00, -2.467447401828151e+00, -1.641502816723996e+00, -1.642979557298576e+00, -4.101904640192358e-01, -4.103073873692123e-01, -2.273091515657790e-01, -1.609613393103387e-02, -8.372965807861432e-02, -5.105180431958154e-04, -1.710850035093980e-02, -1.680302710184064e-02, -3.445960430841684e-04, -2.424175402376939e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_5_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_5", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-4.831507339620774e-04, 0.000000000000000e+00, -4.814735032293648e-04, -1.963380293451493e-03, 0.000000000000000e+00, -1.957631327692739e-03, -7.848592165805698e-02, 0.000000000000000e+00, -8.189045251926674e-02, -7.312721528571903e+00, 0.000000000000000e+00, -3.701954790588323e+01, -9.355134363483066e+01, 0.000000000000000e+00, -9.285234673720881e+04, -2.217204791434497e-01, 0.000000000000000e+00, -3.309694942748272e+01, -4.304376161965334e-01, 0.000000000000000e+00, -4.203582622752144e+05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_5_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_5", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_5_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_5", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.810609740990664e-02, 1.809328152791389e-02, 2.462990731442919e-02, 2.462657051077722e-02, 1.863296484541786e-02, 2.008808566146879e-02, 2.037390723537746e-01, 4.729183999966926e-04, 2.113710755154301e-01, 3.783135698011745e-05, 6.708089672321937e-08, 4.810066528423081e-04, 4.138280636455943e-16, 1.833736959545133e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
