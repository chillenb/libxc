
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_golden_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_golden", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([1.708445482251228e+01, 8.832161653382073e+00, 1.369798820944263e+00, 1.357128942357840e-01, 4.107854193457327e-02, 8.917533116741353e-01, 3.919928163749357e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_golden_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_golden", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([2.520129116934255e+01, 2.524894900195659e+01, 1.155129710032355e+01, 1.157235924075312e+01, -1.926374033483017e-01, -1.960320074784574e-01, 2.103069886752161e-01, -8.821146604654112e-01, 1.326005819637818e-02, -3.496202058459864e-01, -8.748065577186979e-01, -9.053895308591792e-01, -4.097731375441126e-01, -3.425025566712487e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_golden_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_golden", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([5.567447088906173e-03, 0.000000000000000e+00, 5.552424908039286e-03, 1.663520963714600e-02, 0.000000000000000e+00, 1.659280502912859e-02, 1.199873056818140e+00, 0.000000000000000e+00, 1.201405176060155e+00, 7.525649429849428e+00, 0.000000000000000e+00, 2.261945565178679e+04, 1.207994033064801e+02, 0.000000000000000e+00, 7.090421656287625e+08, 1.945226040471822e+04, 0.000000000000000e+00, 1.988294349621030e+04, 2.379330233221575e+09, 0.000000000000000e+00, 6.622368639119661e+09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
