
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b3p86_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3p86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.509890692001683e+00, -1.083653761353919e+00, -3.442661584410233e-01, -1.476816335049517e-01, -7.009253021800974e-02, -1.049459902492281e-01, -3.886247192435321e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b3p86_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3p86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.918364759225273e+00, -1.919890415990906e+00, -1.332658288199264e+00, -1.333617788361417e+00, -3.097020897532736e-01, -3.096182285146276e-01, -1.907896454648468e-01, -1.430760808434812e-01, -7.636166958146873e-02, -7.065534664312743e-02, -3.771238476408995e-02, -3.792394605032970e-02, -5.704072614239887e-03, -5.017560160205725e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b3p86_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b3p86", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.619127200237703e-04, 7.068924656410797e-05, -1.612602752885187e-04, -6.230786810671371e-04, 2.377809787538394e-04, -6.207554294752277e-04, -7.712549745826049e-02, 1.089220072013354e-02, -7.710119972869074e-02, -1.050683044293842e+00, 4.248174684685713e+00, -9.623805677242322e+02, -3.150017267716989e+01, 4.729671416531976e+01, -3.492216012799992e+07, -8.387080557570698e+02, -7.762509200157928e-03, -8.400464483002223e+02, -1.036801610330859e+08, -1.700479803530266e-29, -3.088520081156214e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
