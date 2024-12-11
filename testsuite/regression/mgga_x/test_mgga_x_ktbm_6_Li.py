
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_ktbm_6_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_6", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.883181725417999e+00, -1.259934473284701e+00, -2.957040098365474e-01, -1.729326885961596e-01, -6.089900804411837e-02, -1.335759819344021e-02, -2.463844634036363e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_ktbm_6_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_6", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.635148441006938e+00, -2.637707341737070e+00, -1.816247075549248e+00, -1.817919323375784e+00, -3.661000379089703e-01, -3.651755953221590e-01, -2.381353378466926e-01, -1.592363593594349e-02, -7.648509873451384e-02, -5.050397505634467e-04, -1.674528800634998e-02, -1.662296820700913e-02, -3.373375675950843e-04, -2.406607011077929e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_6_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_6", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.039157033582805e-04, 0.000000000000000e+00, -5.021829363472570e-04, -1.950128795845519e-03, 0.000000000000000e+00, -1.944594056488054e-03, -5.468114000094926e-02, 0.000000000000000e+00, -5.684811111173906e-02, -7.804729606251307e+00, 0.000000000000000e+00, -2.754183785007728e+01, -7.512082249044097e+01, 0.000000000000000e+00, -6.904147576710567e+04, -5.125960242688868e-01, 0.000000000000000e+00, -2.462435555347087e+01, -1.045487486552378e+00, 0.000000000000000e+00, -1.252811527359137e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_ktbm_6_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_ktbm_6", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([1.788809933209980e-02, 1.787707975443094e-02, 2.005552224539816e-02, 2.006693354044451e-02, -5.183769052639311e-03, -5.353120856636842e-03, 2.130110478164837e-01, 3.519138575820240e-04, -3.946676230998587e-02, 2.812997443353677e-05, 7.612730380548963e-06, 3.579495122996805e-04, 1.269389654791425e-10, -1.212809052791654e-11])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
