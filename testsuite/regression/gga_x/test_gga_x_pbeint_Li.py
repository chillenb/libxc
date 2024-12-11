
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_pbeint_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.770282867310317e+00, -1.252985734009023e+00, -4.005040159911846e-01, -1.587027331996041e-01, -7.646046868558441e-02, -2.054447744523240e-02, -3.838586978687961e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_pbeint_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.268510002401344e+00, -2.270635571534970e+00, -1.544167721466548e+00, -1.545551811540085e+00, -3.571140880348505e-01, -3.572720299892732e-01, -2.068717257459846e-01, -2.611568089395331e-02, -7.121283736973040e-02, -8.296433205348372e-04, -2.745717047315166e-02, -2.725988296407768e-02, -5.541555286516825e-04, -3.939542015791528e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_pbeint_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_pbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.556708316893641e-04, 0.000000000000000e+00, -1.551131428732809e-04, -6.627826612646331e-04, 0.000000000000000e+00, -6.605519869422120e-04, -8.577644074784262e-02, 0.000000000000000e+00, -8.560787280630011e-02, -2.332489364171363e+00, 0.000000000000000e+00, -2.781341492275524e-01, -6.725417917827930e+01, 0.000000000000000e+00, -1.776459486577152e+00, -2.826921493316097e-01, 0.000000000000000e+00, -2.639635918990545e-01, -1.293193943886323e+00, 0.000000000000000e+00, -1.851073014209581e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
