
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbe_jsjr_Li_restr_1_zk():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_jsjr", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.776825500217467e+00, -1.260685962581696e+00, -3.944813977119921e-01, -1.273095417597733e-01, -6.473536716237702e-02, -2.053838151863960e-02, -3.654155879996823e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbe_jsjr_Li_restr_1_vrho():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_jsjr", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.263777659995150e+00, -1.541805830267473e+00, -3.723124780386641e-01, -1.626780490411188e-01, -6.120434737063494e-02, -2.733675941375319e-02, -4.872201710204106e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbe_jsjr_Li_restr_1_vsigma():
    # Prepare the input
    inp = test_data["Li_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbe_jsjr", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.011989640959753e-05, -3.658704287236179e-04, -3.723633408005497e-02, -3.412232221691948e+00, -5.494531088899733e+01, -1.976003137520660e-01, -1.027998588764516e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
