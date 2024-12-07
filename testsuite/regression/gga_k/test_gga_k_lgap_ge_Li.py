
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_lgap_ge_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lgap_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.644442056157229e+01, 8.196899316913104e+00, 9.161343562278486e-01, 1.334467991712174e-01, 3.037073071595438e-02, 2.867665736307356e+00, 3.983846988792857e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_lgap_ge_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lgap_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [2.591666383970519e+01, 2.596438751066064e+01, 1.221337637178284e+01, 1.223484987004507e+01, 1.358679132508521e-01, 1.329728711971319e-01, 2.138231279363904e-01, -6.425124490180711e+00, 2.230657651626990e-02, -4.793566903667206e+01, -6.053142880218191e+00, -6.410875588065718e+00, -9.096447397357758e+01, -9.773964681631351e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_lgap_ge_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_lgap_ge", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.513550730420077e-03, 0.000000000000000e+00, 2.506776052942429e-03, 7.570642971051244e-03, 0.000000000000000e+00, 7.551127717894203e-03, 6.742408473558261e-01, 0.000000000000000e+00, 6.754799121536241e-01, 3.419077805829369e+00, 0.000000000000000e+00, 1.089233433629554e+05, 6.193893963804403e+01, 0.000000000000000e+00, 6.259278175164235e+10, 8.911113236495637e+04, 0.000000000000000e+00, 9.314888483910087e+04, 3.398717851023867e+11, 0.000000000000000e+00, 1.215793249261282e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
