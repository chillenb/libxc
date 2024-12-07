
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_th1_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.935098106703837e+00, -1.336822998301763e+00, -4.700600305396275e-01, -1.619695945342129e-01, -7.284680501723481e-02, -1.469449017325280e-01, 4.005764647767726e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_th1_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.665853800892711e+00, -2.663488367015609e+00, -1.733070970189967e+00, -1.733515465875525e+00, -2.711962915594823e-01, -2.709874220407929e-01, -2.085811058995777e-01, -6.036282995200720e-02, -7.707395411264190e-02, 1.307606665924035e-01, 7.189652814965226e-02, 7.670684903495495e-02, -3.143368833682301e-01, -5.041788887159434e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_th1_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th1", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.711454311790345e-04, -1.107313139846228e-03, 6.700481507051170e-04, 6.257899593950732e-04, -1.732646253339628e-03, 6.229070418418056e-04, -8.142257695722992e-02, -1.873315696919617e-01, -8.142125806966981e-02, -4.397722817782021e+00, -5.343829980087684e+00, -8.410329360665482e+02, -5.540350460015183e+01, -1.120703553448702e+02, -7.818221526520129e+07, -5.566173094195860e+02, -4.498219863257240e+03, -5.640217450803938e+02, 1.261776302032519e+08, 8.034111774576547e+09, -5.862026535430782e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
