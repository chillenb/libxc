
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_pkzb_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.749708557698871e+00, -1.209129191267948e+00, -3.886042309500469e-01, -1.584900330668411e-01, -6.737859113872846e-02, -2.055685952484587e-02, -3.838588870219928e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_pkzb_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.347926270158240e+00, -2.350129907641539e+00, -1.625936561360765e+00, -1.627354385230859e+00, -3.005509213462756e-01, -2.971336545035437e-01, -2.123959501978878e-01, -2.615897632319181e-02, -6.828298347640380e-02, -8.296468243197490e-04, -2.750809929932611e-02, -2.730785489448631e-02, -5.541564195188991e-04, -3.939545845218585e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_pkzb_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-2.321449671796331e-04, 0.000000000000000e+00, -2.313107485865396e-04, -8.838440768170262e-04, 0.000000000000000e+00, -8.814253910007455e-04, 2.300425208864543e-01, 0.000000000000000e+00, 2.331671102554485e-01, -3.613739300402852e+00, 0.000000000000000e+00, -1.624110715563185e-03, 5.809462811531144e+01, 0.000000000000000e+00, -2.638441977200220e-05, -1.576943956305098e-08, 0.000000000000000e+00, -1.636325606617371e-03, -3.629902670669122e-21, 0.000000000000000e+00, 7.071321293359217e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_pkzb_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_pkzb", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.302378300141406e-03, 2.300319891086313e-03, 2.114719328454316e-03, 2.118932865766001e-03, -3.993633405153985e-02, -4.075516077976808e-02, 2.885629490700687e-02, 8.678093643874245e-09, -1.804806262454614e-01, 4.498395761455999e-15, 9.800009695291063e-14, 9.946590835096205e-09, 1.844262206348943e-31, -1.735164512991694e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
