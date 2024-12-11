
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_mn12_sx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.861995873194692e-01, -7.908664816437638e-01, 2.604827580024616e-01, -1.409619252304181e-01, 2.821663383737058e-02, 9.756266032568614e-03, 4.069396207847585e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_mn12_sx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([9.994718010019266e-01, 1.008703831863205e+00, -8.166406868406481e-01, -8.142209482087641e-01, -4.176395224263199e-01, -4.312191748647711e-01, -3.734070406248117e-01, 1.232581618437108e-02, -1.358351302605448e-01, 4.569240305030753e-04, 1.247229476537122e-02, 1.278189885339711e-02, 3.057751851796208e-04, 1.200753746352558e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mn12_sx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-4.475048896345274e-03, 0.000000000000000e+00, -4.470547073958005e-03, -6.582044729544339e-03, 0.000000000000000e+00, -6.590011414198368e-03, -3.029124754949672e+00, 0.000000000000000e+00, -2.986996727565647e+00, 3.712235072086967e+01, 0.000000000000000e+00, -3.665639145235511e+00, -1.364807532083352e+03, 0.000000000000000e+00, -2.345673416370314e+01, -1.575508570965661e-03, 0.000000000000000e+00, -3.478179461790459e+00, -1.070620326661151e-09, 0.000000000000000e+00, -7.480883672336636e+13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_mn12_sx_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_mn12_sx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-2.238544475917271e-01, -2.250223804788146e-01, -2.068414639235962e-02, -2.106222958195020e-02, 1.377493067060756e-01, 1.408849060513570e-01, 5.286417868683196e+00, 2.438923148147088e-05, 1.381596374969752e+00, 5.199432122154035e-09, 1.202805735248753e-08, 2.627959050266486e-05, 7.076141434248811e-20, 4.986583631208689e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
