
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_gas22_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_gas22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.605548607870510e+00, -9.963196970868249e-01, -1.998947703437708e-01, 5.017766640369382e-02, -1.045340475199775e-02, -1.474818710591447e-01, -3.197956570586095e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_gas22_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_gas22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.391014899454584e+00, -2.384847340730384e+00, -1.557324785786492e+00, -1.558316778710080e+00, -3.115607029432377e-01, -3.129172296301713e-01, 1.078800047794012e-01, -1.450728424672001e-01, -2.664052593611583e-02, 1.339596311517455e+00, 2.679067880843937e-02, -1.825984269800577e-01, 1.550952237720839e-01, -9.675842651827241e-01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_gas22_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_gas22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([8.967719763921661e-03, 0.000000000000000e+00, 8.887635690733361e-03, 3.212646902069091e+20, 0.000000000000000e+00, 3.190855082246712e+20, 4.713871361459711e+20, 0.000000000000000e+00, 4.729939506224056e+20, -6.331253205094085e+01, 0.000000000000000e+00, 1.974923165310748e+04, -3.859573563809875e+03, 0.000000000000000e+00, 1.003966842704153e+10, -3.622447779773120e+01, 0.000000000000000e+00, -3.843837400781322e+01, -6.650854441800662e+03, 0.000000000000000e+00, -2.371320345912453e+14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_gas22_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_gas22", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([2.556042133428575e-02, 2.556175396507953e-02, 2.633473902394819e-02, 2.632458630303043e-02, 8.097194226477618e-03, 8.365082668246296e-03, -2.265015417398673e+00, -3.792359382607870e-05, 9.859864285603637e-02, -1.747323318470447e-06, -2.716083609357980e-07, -6.105818981702921e-04, -1.336139807678505e-16, -4.289145764035828e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
