
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_xc_corrksdt_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_corrksdt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.823552518556801e+00, -1.284773088203150e+00, -3.387468890046067e-01, -1.749552528185666e-01, -7.322549592417335e-02, -1.820342299305641e-02, -3.744627435232740e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_xc_corrksdt_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_xc_corrksdt", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.408683730935335e+00, -2.410607798482232e+00, -1.693188547051227e+00, -1.694415664032859e+00, -4.419499236649025e-01, -4.418303458522292e-01, -2.301484025213031e-01, -1.530634863538315e-01, -9.613112976770023e-02, -7.848944926621816e-02, -2.382039226797225e-02, -2.381088485648964e-02, -4.927974246160027e-04, -5.105204105708558e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
