
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_kcisk_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.343236635995592e-02, -8.371515710326172e-02, -4.959817863632871e-02, -1.808224134175908e-02, -1.095909861488205e-02, 2.098628511741550e-06, -1.643726357407228e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_kcisk_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.026837581306614e-01, -1.024837253468584e-01, -9.256356629933911e-02, -9.238935586969717e-02, -5.664197430173448e-02, -5.669262374811559e-02, -2.101473789966307e-02, -1.127518937139935e-01, -1.310471645136509e-02, -6.792553316105139e-02, -6.520150866039241e-06, 1.406421871022878e-05, -5.261616749615765e-13, -9.754474556692027e-13])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_kcisk_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.339956725687946e-04, 2.530018784360802e-04, 3.343305789231798e-04, 1.071858795523291e-03, 1.000419278745930e-03, 1.071008903229574e-03, 1.095116165499284e-01, 1.588147563859290e-01, 1.103135904517133e-01, 2.434402725576555e+01, 4.097049596729939e+00, 2.049089673997275e+00, 1.444737346708762e+02, 1.610486248423593e+02, 8.053089457816411e+01, 8.406092813284489e-04, 1.681208510724927e-03, 1.367049460356910e-03, 6.455592911137991e-09, 1.291118581827277e-08, 1.050300266018571e+06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_kcisk_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_kcisk", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.518973902649667e-05, -3.517051707674886e-43, -2.978197219499551e-42, -2.968540450469716e-42, -1.405766216307531e-38, -1.481935180641130e-38, -1.022608013947658e-32, -2.443810853658494e-06, -1.385656088160181e-31, -1.572871022279231e-08, -1.151971137230486e-09, -2.527468185318665e-06, -3.203490006160380e-19, -7.541133187092824e-21])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
