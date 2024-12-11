
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b3lyp_mcm2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp_mcm2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.505265109108313e+00, -1.084640760345599e+00, -3.257633636573516e-01, -1.300621406564171e-01, -6.456534436507058e-02, -1.002001376591727e-01, -3.913924693103873e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b3lyp_mcm2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp_mcm2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.879769048988180e+00, -1.881317610886241e+00, -1.301308489283402e+00, -1.302246579715716e+00, -3.777495935906595e-01, -3.780458892137653e-01, -1.667020864688201e-01, -1.068331922613555e-01, -6.049065251814383e-02, -3.552540208084665e-02, -3.075011287833672e-02, -3.095679128150483e-02, -5.496001659932294e-03, -4.820881775730148e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b3lyp_mcm2_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3lyp_mcm2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.935038628045534e-04, 4.920414408926497e-06, -1.929601516214459e-04, -7.130573590734915e-04, 3.435783859651094e-05, -7.113214661654573e-04, -5.099694219907445e-02, 4.497361993784547e-02, -5.082311716863359e-02, -3.214850199738157e+00, 4.330018566301709e+00, -9.733120866089823e+02, -5.583788432953305e+01, 2.220472923711974e+01, -3.535869442001270e+07, -8.491494736829613e+02, 7.476597286846731e-02, -8.505043983495608e+02, -1.049761630459995e+08, 0.000000000000000e+00, -3.127126582170668e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
