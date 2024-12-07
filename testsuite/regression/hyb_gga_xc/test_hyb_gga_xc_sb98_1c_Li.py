
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_sb98_1c_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_1c", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.518706791157645e+00, -1.087920020445172e+00, -3.617653195061342e-01, -1.422699779431706e-01, -6.834042625491678e-02, -1.393602877045109e-02, -2.712755042298165e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_sb98_1c_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_1c", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.936479192580184e+00, -1.938053642068116e+00, -1.332985516948278e+00, -1.333995635504180e+00, -3.203359152485807e-01, -3.208475843224075e-01, -1.866458430326668e-01, 3.314860561925917e-01, -5.538315192393489e-02, 2.182888839291629e-01, -1.936458529245463e-02, -1.878100050260204e-02, -5.219718165277564e-04, 8.291780424540631e-05]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_sb98_1c_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_sb98_1c", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.191717304550109e-04, 0.000000000000000e+00, -1.187336090914199e-04, -5.417686034298845e-04, 0.000000000000000e+00, -5.398296304571112e-04, -7.912201919663048e-02, 0.000000000000000e+00, -7.885843331593138e-02, -4.728725221499793e-01, 0.000000000000000e+00, 5.112745675197144e+01, -7.763077201052002e+01, 0.000000000000000e+00, 6.112568473173108e+03, -1.666469860884083e-01, 0.000000000000000e+00, -7.599588390683727e-02, -2.584805870095339e+00, 0.000000000000000e+00, 1.097822261910981e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
