
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lc_wpbe_whs_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbe_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.634470074960733e+00, -1.108180088889975e+00, -2.152325109088349e-01, -4.944949039205360e-02, -4.359898502107795e-03, -1.814502242425370e-05, -1.237640080062206e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lc_wpbe_whs_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbe_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.143721708078390e+00, -2.145720357341409e+00, -1.409683836742340e+00, -1.410946758774077e+00, -2.166808577472015e-01, -2.168294572789503e-01, -8.512948183813969e-02, -9.767213074559540e-02, -1.172933532405082e-02, 3.428185822405140e-01, -3.684793025819380e-05, -3.604247089982593e-05, -2.979998255755342e-10, -1.070694355664493e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lc_wpbe_whs_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_wpbe_whs", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.956691539963190e-04, 9.190971700708733e-05, -1.948413891104439e-04, -7.991337684939129e-04, 2.980993506782570e-04, -7.961362142636252e-04, -6.344198385668547e-02, 6.249948659585063e-03, -6.328615277788621e-02, 2.850371610286359e+00, 6.762268918356340e+00, 3.381125101542766e+00, 9.202903709330961e+00, 2.258698854598489e+01, 1.129349427298224e+01, 1.553039190783271e-04, 3.357174600576258e-04, 1.565965507172214e-04, 1.606542114058703e-06, 3.212885779437900e-06, 1.606543051037621e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
