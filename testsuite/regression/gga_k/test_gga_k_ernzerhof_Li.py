
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_ernzerhof_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ernzerhof", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.634793891068669e+01, 8.129158080814108e+00, 1.051504125276346e+00, 1.328911139897722e-01, 3.119220509490080e-02, 3.040333903700108e+00, 1.356881421936126e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_ernzerhof_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ernzerhof", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.590259083818314e+01, 2.595032101860689e+01, 1.218559577666516e+01, 1.220710276347770e+01, -4.883110737693372e-01, -4.954190770574931e-01, 2.137415414247118e-01, -3.122522693553492e+00, 1.587950006849871e-02, -1.210295932801763e+00, -3.104199090487947e+00, -3.209112427978389e+00, -1.418477664605027e+00, -1.185602041107430e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_ernzerhof_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_ernzerhof", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.262219506856761e-03, 0.000000000000000e+00, 2.255897092168860e-03, 7.122537780932779e-03, 0.000000000000000e+00, 7.103543971618601e-03, 1.085997512706073e+00, 0.000000000000000e+00, 1.089470169597320e+00, 2.989794131677767e+00, 0.000000000000000e+00, 7.828236377612413e+04, 7.901090249764842e+01, 0.000000000000000e+00, 2.454376723890904e+09, 6.731796800930204e+04, 0.000000000000000e+00, 6.880999563696797e+04, 8.236143113400122e+09, 0.000000000000000e+00, 2.292358374917305e+10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
