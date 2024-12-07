
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_bloc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.968927759386149e+00, -1.367221933918863e+00, -4.000392256019930e-01, -1.770358986477462e-01, -7.665000837653904e-02, -2.054447947414146e-02, -3.838586978645408e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_bloc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.476265977286818e+00, -2.478542319650106e+00, -1.777160634100805e+00, -1.779234396433515e+00, -3.554884806269564e-01, -3.559260256405889e-01, -2.171290966375743e-01, -2.611569208058222e-02, -7.716926578353402e-02, -8.296433204579740e-04, -2.745718545517738e-02, -2.725989618662414e-02, -5.541555286317482e-04, -3.939542015705412e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_bloc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.741776670019674e-04, 0.000000000000000e+00, -6.736578433197708e-04, -5.886029292078965e-04, 0.000000000000000e+00, -5.834833052046659e-04, -8.454648125760811e-02, 0.000000000000000e+00, -8.357557332380661e-02, -2.372532532543081e+01, 0.000000000000000e+00, -2.780506390221201e-01, -4.852369624049394e+01, 0.000000000000000e+00, -1.776498264492767e+00, -2.825911408736027e-01, 0.000000000000000e+00, -2.638791256248716e-01, -1.293222808933014e+00, 0.000000000000000e+00, -1.851114589210360e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_bloc_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_bloc_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_bloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [3.493593308440483e-02, 3.503464963517926e-02, 9.755818851474082e-03, 9.737975209460589e-03, -6.514315855681271e-04, -9.089400431878626e-04, 8.469552045749745e-01, 6.652326229178553e-11, -2.391261446717664e-02, 3.628420903174772e-17, 5.821712304763515e-20, 7.620791406752818e-11, 6.904956655432262e-42, 6.041050894501508e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
