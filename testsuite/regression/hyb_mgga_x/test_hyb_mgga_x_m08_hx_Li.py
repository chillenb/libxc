
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m08_hx_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-5.302963570617911e-01, -5.328859702114425e-01, -7.848416214546768e-02, -3.496208636552966e-02, -2.479242455039819e-02, 8.798127751367169e-03, 1.366823628874164e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m08_hx_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.050261892373087e-02, -3.970162825835841e-02, -6.027294643849199e-01, -6.011083252644687e-01, -1.967222460778280e-01, -2.008935293480430e-01, -1.201149692325383e-02, 1.068243821847402e-02, -4.021271438057753e-02, 3.598538948688970e-04, 1.191714831995219e-02, 1.110918360245439e-02, 2.403978967765259e-04, 2.038104131944629e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m08_hx_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.793766169743650e-04, 0.000000000000000e+00, -5.847685906705545e-04, 7.059213768270758e-04, 0.000000000000000e+00, 7.078105776537472e-04, -6.613579625908049e-01, 0.000000000000000e+00, -6.588872129830088e-01, -4.466580382966424e+01, 0.000000000000000e+00, 6.894857210472789e-01, -2.650624211768690e+02, 0.000000000000000e+00, 4.495561503194167e+00, 2.993383535583435e-04, 0.000000000000000e+00, 6.535181283415901e-01, 2.052039370626784e-10, 0.000000000000000e+00, 4.061872095821158e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m08_hx_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m08_hx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-9.477399496460352e-02, -9.488547844580987e-02, -1.572885079590312e-02, -1.603524574649559e-02, 1.688976661071021e-02, 1.776442119484798e-02, -9.850271869788481e-01, 6.065748064944157e-05, 5.991815920960070e-02, 1.247137862616233e-08, 3.027583920702895e-08, 6.545354254684051e-05, 1.694525324047923e-19, 2.641201659853477e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
