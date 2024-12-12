
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_18_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-2.173270022657789e+00, -1.531069700441768e+00, -2.771365340584860e-01, -1.937873185996120e-01, -6.630482555513881e-02, -9.803074428448362e-03, -1.835039426697459e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_18_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.621445910250648e+00, -2.623969332766020e+00, -1.771060536154856e+00, -1.772286710705109e+00, -3.685802681430238e-01, -3.708093052437639e-01, -2.428659123776196e-01, -1.190902798788981e-02, -9.023348760589911e-02, -3.776212803176268e-04, -1.316147684787765e-02, -1.243222993787274e-02, -2.652472963055304e-04, -1.793119340325329e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_18_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-8.616335457746717e-04, 0.000000000000000e+00, -8.587149587645591e-04, -3.376457995842853e-03, 0.000000000000000e+00, -3.367666412218391e-03, -6.143975800746801e-02, 0.000000000000000e+00, -6.516904104201922e-02, -1.314145317968257e+01, 0.000000000000000e+00, -1.382019541211898e+01, -9.745817153068532e+01, 0.000000000000000e+00, -3.456522319334855e+04, 3.209274018911378e-01, 0.000000000000000e+00, -1.235797432305567e+01, 6.831591474765470e-01, 0.000000000000000e+00, -1.564816837584095e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_18_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([3.240096430235893e-02, 3.235963026399521e-02, 5.432238166481317e-02, 5.428180153518578e-02, 2.308030952095644e-02, 2.506775677517080e-02, 3.245705871561517e-01, 1.770170856524248e-04, 3.882078885434970e-01, 1.408319937506523e-05, -9.758537009135187e-08, 1.801053362153173e-04, -6.568007334729278e-16, 6.826241278256534e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
