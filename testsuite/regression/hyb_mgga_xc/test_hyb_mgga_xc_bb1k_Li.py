
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_xc_bb1k_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_bb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.066987340449006e+00, -7.639351423048065e-01, -3.706199493501207e-01, -5.019423412078206e+79, -6.711725221994821e+80, -2.155879321669569e+109, -1.344435890261890e+56]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_xc_bb1k_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_bb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.387659073288405e+00, -1.388712220580933e+00, -9.669508167211386e-01, -9.676204510534366e-01, -1.593936177368635e-01, -1.616196733355546e-01, 2.792852842238862e+66, -9.848142559886196e+80, 1.078569452337341e+71, -2.409368518039024e+85, -1.074057739152810e+107, -1.097595813315504e+107, -1.670484991869728e+54, -4.648373983428211e+40]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_bb1k_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_bb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-9.193557425363024e-05, 0.000000000000000e+00, -9.176729645659364e-05, -2.934061607462544e-05, 0.000000000000000e+00, -3.106344910860318e-05, 4.699236701811946e+00, 0.000000000000000e+00, 4.595741259982414e+00, 2.251068842847621e+00, 0.000000000000000e+00, 5.649063800257172e+00, 2.294614754389186e+03, 0.000000000000000e+00, 6.632508543915026e+02, 1.193814816847380e-04, 0.000000000000000e+00, 8.771345115992075e-03, 5.497290808982469e-11, 0.000000000000000e+00, 3.225550180688173e+20]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_xc_bb1k_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_xc_bb1k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-2.873107274983154e-03, -2.871683944547766e-03, -5.332155022208856e-03, -5.324538201822170e-03, -5.428445283005290e-02, -5.432066785617735e-02, -1.392078133987678e-01, -1.176431803685671e-07, -5.378184861852190e-01, -3.855053355621430e-11, -5.743566782136254e-11, -1.256638198788993e-07, -5.361832595959912e-22, -6.047408678635331e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
