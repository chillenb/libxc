
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbefe_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbefe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.820466700063890e-02, -5.239030869084750e-02, -5.891699545120296e-03, -1.613235679913936e-02, -2.405435074225741e-03, -1.831190967523794e-08, -4.324882466044677e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbefe_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbefe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.173674552032603e-01, -1.172304486204500e-01, -1.058549234073060e-01, -1.057476619695564e-01, -2.748267438603019e-02, -2.749279070383471e-02, -2.332332690210642e-02, -1.073509151300814e-01, -1.023142598074341e-02, 4.865823486559687e-01, -1.184834815284965e-07, -1.190787173821585e-07, -2.736045168639373e-15, -3.237461568359606e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbefe_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbefe", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.917157238112952e-05, 7.834314476225904e-05, 3.917157238112952e-05, 1.346948796500921e-04, 2.693897593001841e-04, 1.346948796500921e-04, 5.236774561872957e-03, 1.047354912374592e-02, 5.236774561872957e-03, 2.349305905923026e+00, 4.698611811846050e+00, 2.349305905923026e+00, 1.673712347058066e+01, 3.347424694116131e+01, 1.673712347058066e+01, 4.038086792797140e-04, 8.076173585740904e-04, 4.038086792797140e-04, 3.868406268587755e-06, 7.737120095121081e-06, 3.868406268587755e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
