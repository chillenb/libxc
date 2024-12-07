
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_gdme_nv_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gdme_nv", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.467657737048269e+01, -1.271797086510167e+00, 1.202297303755625e-01, -1.733417610960458e-01, -1.506334539131612e-02, 4.795631080231802e+02, 4.226094874618420e+07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_gdme_nv_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gdme_nv", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.003715402151582e+01, -1.003549637318535e+01, -3.146411721853391e+00, -3.148608391355902e+00, -9.011787377492779e-01, -8.938250779188079e-01, -4.067350698218280e-01, -1.539464152477372e+00, -1.793143527728578e-01, -9.399132051510897e+01, -3.144355024599847e+02, -1.919004520772829e+00, -1.914817250956921e+07, -2.190137553496455e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gdme_nv_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gdme_nv", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gdme_nv_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gdme_nv", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [-1.064331943166730e-02, -1.063373814691312e-02, -1.532972943877615e-02, -1.531669274709615e-02, -6.381216765768456e-02, -6.383931671471478e-02, -1.176810112364985e-01, -1.698318394692914e+00, -2.968565665947828e-01, -5.354871881626720e+01, -1.615034318955743e+00, -1.626866650570119e+00, -8.016964702567438e+01, -1.127706753399200e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gdme_nv_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gdme_nv", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [8.514655545333838e-02, 8.506990517530497e-02, 1.226378355102092e-01, 1.225335419767692e-01, 5.104973412614765e-01, 5.107145337177182e-01, 9.414480898919881e-01, 1.358654715754331e+01, 2.374852532758263e+00, 4.283897505301376e+02, 1.292027455164595e+01, 1.301493320456095e+01, 6.413571762053950e+02, 9.021654027193597e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
