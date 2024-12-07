
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_sg4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sg4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.800599950250208e+00, -1.287282114155849e+00, -3.983443701992958e-01, -1.604500394029768e-01, -7.764139326757738e-02, -2.055572349606353e-02, -3.838588695208896e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_sg4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sg4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.245450625040793e+00, -2.247558404426554e+00, -1.536358944688208e+00, -1.537700550977454e+00, -3.639174078306122e-01, -3.636934368192349e-01, -2.051372918353157e-01, -2.615510535288465e-02, -7.970146203409194e-02, -8.296465001647618e-04, -2.750338622960417e-02, -2.730357504183007e-02, -5.541563370926831e-04, -3.939545490910701e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_sg4_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sg4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.644723502081223e-04, 0.000000000000000e+00, -2.636171787844398e-04, -9.448145343799909e-04, 0.000000000000000e+00, -9.419787282905708e-04, -8.108375604969000e-02, 0.000000000000000e+00, -8.109990601577799e-02, -4.310418348709883e+00, 0.000000000000000e+00, -2.573996859333219e-02, -5.212460989971229e+01, 0.000000000000000e+00, -1.643644377682709e-01, -2.616257539616058e-02, 0.000000000000000e+00, -2.442894327099764e-02, -1.196509257225910e-01, 0.000000000000000e+00, -1.712678768603778e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
