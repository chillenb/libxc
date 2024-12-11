
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_b88_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.621597218845438e-02, -5.779821309743782e-02, -1.242859003931817e-01, -1.643733878539518e-03, -1.746796153344852e-02, -9.482258564519751e-05, -4.022888232979444e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_b88_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-4.973533520428060e-02, -4.959921356784270e-02, -4.345456363878107e-02, -4.337799304421693e-02, 1.849111331577167e-02, 1.724392510736829e-02, 6.179248613556778e-04, -3.882742529238692e-01, 7.164737623460782e-03, -3.368791850331554e+00, -1.376531351942599e-01, -3.610078983135199e-02, -1.303626492712442e-01, 5.085371622446052e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_b88_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [5.789302306854838e-05, 0.000000000000000e+00, 5.755851177938902e-05, 3.618842171174339e-04, 0.000000000000000e+00, 3.597569265769828e-04, 8.328326624198155e-01, 0.000000000000000e+00, 8.197111812713526e-01, 2.096906673273303e+00, 0.000000000000000e+00, 3.011502254329589e+03, 3.247591951999253e+02, 0.000000000000000e+00, 1.867073683445849e+09, 6.242734140015302e+01, 0.000000000000000e+00, 7.881806915466435e+02, 5.993006652247173e+03, 0.000000000000000e+00, 1.453173509344476e+19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_b88_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_b88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-1.422695299083859e-03, -1.420706577777779e-03, -2.770134907194253e-03, -2.765896738210295e-03, -2.989585738318108e-02, -2.991519116320938e-02, -7.280347473382269e-02, -1.161169913169434e-02, -2.437582623813458e-01, -9.334848938795650e-03, -9.269570312774266e-04, -1.143978317059454e-02, -7.276469707445078e-07, -1.283731563084314e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
