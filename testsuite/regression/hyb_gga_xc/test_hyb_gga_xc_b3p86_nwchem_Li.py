
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b3p86_nwchem_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3p86_nwchem", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.527977704218032e+00, -1.101461478882408e+00, -3.589042305102618e-01, -1.611480381661582e-01, -8.095837298323875e-02, -1.108151049070181e-01, -3.918684999314774e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b3p86_nwchem_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3p86_nwchem", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.936647735828102e+00, -1.938172459920320e+00, -1.350790494439714e+00, -1.351749249489727e+00, -3.251821689136598e-01, -3.250966261136194e-01, -2.051049486665157e-01, -1.526778057979243e-01, -8.824123522215103e-02, -7.897185423158927e-02, -4.455975942183199e-02, -4.475395871068778e-02, -6.164515875021165e-03, -5.326371770760344e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b3p86_nwchem_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3p86_nwchem", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.619127200237703e-04, 7.068924656410797e-05, -1.612602752885187e-04, -6.230786810671371e-04, 2.377809787538394e-04, -6.207554294752277e-04, -7.712549745826049e-02, 1.089220072013354e-02, -7.710119972869074e-02, -1.050683044293842e+00, 4.248174684685713e+00, -9.623805677242322e+02, -3.150017267716989e+01, 4.729671416531976e+01, -3.492216012799992e+07, -8.387080557570698e+02, -7.762509200157928e-03, -8.400464483002223e+02, -1.036801610330859e+08, -1.700479803530266e-29, -3.088520081156214e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
