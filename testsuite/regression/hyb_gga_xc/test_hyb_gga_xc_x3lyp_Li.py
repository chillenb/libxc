
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_x3lyp_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_x3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.467796558975385e+00, -1.059806275473827e+00, -3.218134913423228e-01, -1.298066375461907e-01, -6.517705248813578e-02, -7.674860831852823e-02, -2.919436201037794e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_x3lyp_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_x3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.830641398952383e+00, -1.832123749594400e+00, -1.269707383536714e+00, -1.270602373140490e+00, -3.772027317270928e-01, -3.775232149561275e-01, -1.657042837476875e-01, -1.084714845175001e-01, -6.209588414009629e-02, -4.069562717607073e-02, -2.574694252231921e-02, -2.591103747804200e-02, -4.186457529768259e-03, -3.683024198955669e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_x3lyp_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_x3lyp", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.882585169154883e-04, 4.549072232432841e-06, -1.877253510531885e-04, -6.934080025322302e-04, 3.176486298435520e-05, -6.916986682654063e-04, -4.612466028978465e-02, 4.157947454183569e-02, -4.594030599290373e-02, -3.124848782618467e+00, 4.003233384193598e+00, -7.223559275655481e+02, -5.295349650041919e+01, 2.052894508601134e+01, -2.630729521199071e+07, -6.305311363770155e+02, 6.912340767268339e-02, -6.316238080226766e+02, -7.810355626011799e+07, 0.000000000000000e+00, -2.326620711465969e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
