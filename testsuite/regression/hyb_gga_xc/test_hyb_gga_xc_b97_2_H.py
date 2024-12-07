
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b97_2_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.327450826582470e-01, -4.732735907064607e-01, -2.863494923079620e-01, -1.434079395657727e-01, -1.125529351921693e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b97_2_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-7.083836230349841e-01, -2.611342429490208e-01, -6.210927059074131e-01, -2.585190019099186e-01, -3.403199096829130e-01, -2.079781289866571e-01, -6.714285245652335e-02, 6.315357478520842e-02, -1.494613136875444e-02, 3.541976698779288e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b97_2_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.919908140598221e-02, 0.000000000000000e+00, -2.278614201597911e+20, -3.717072755254811e-03, 0.000000000000000e+00, -1.377316755806567e+20, -8.758691738927823e-02, 0.000000000000000e+00, -4.802311967773370e+18, -1.390089007995715e+01, 0.000000000000000e+00, 7.398917881520580e+19, -2.759056215490180e+01, 0.000000000000000e+00, 8.560919864913514e+13]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
