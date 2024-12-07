
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_optc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_optc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.880406006524254e-02, -4.493011075647285e-02, -4.147859216339926e-03, -1.008617603171354e-02, -9.618127524879686e-04, -8.166063620299009e-09, -1.714394735606090e-16]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_optc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_optc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.015474636214298e-01, -1.013977461017569e-01, -9.067739449345344e-02, -9.056174624060537e-02, -2.040504578516930e-02, -2.042167117518657e-02, -1.596402293824255e-02, -1.068739480664332e-01, -4.613763730432028e-03, 4.466428842135798e-01, -5.278345549596736e-08, -5.337695083931873e-08, -1.047502991065403e-15, -1.424594558763026e-15]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_optc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_optc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.385523100159167e-05, 8.776009744301140e-05, 2.390991226646024e-05, 8.414187028302920e-05, 2.880880988566203e-04, 8.431530229311079e-05, 3.616764864758909e-03, 8.350371939120475e-03, 3.621132566497787e-03, 2.165922382065001e+00, 7.183903905319476e+00, 3.591931444220927e+00, 7.831481815775589e+00, 2.604278928801110e+01, 1.302139343755617e+01, 1.682710691242311e-04, 3.815690560284414e-04, 1.706730137663303e-04, 1.413426948559105e-06, 3.540118209206232e-06, 1.447062930953570e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
