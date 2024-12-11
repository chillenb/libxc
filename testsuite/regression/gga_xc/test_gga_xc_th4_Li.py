
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_th4_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.899764365340251e+00, -1.348608941681480e+00, -4.486087254673451e-01, -1.673348342344087e-01, -5.978147101214156e-02, -2.202104944275709e-01, -4.024754001697234e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_th4_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.492760396435371e+00, -2.498725071322589e+00, -1.679992899110686e+00, -1.682058204549492e+00, -3.013463721317570e-01, -3.010979612669175e-01, -2.203483607621505e-01, -1.885983370821680e-02, -7.890906194926633e-02, -5.764150009987648e-02, 1.026589022344329e-01, 1.055842673876302e-01, 5.318686083932433e-04, 1.459054995627601e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_th4_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_th4", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-5.153445090917518e-04, 9.704567334027229e-04, -5.161047903137357e-04, -1.671702559284659e-03, 2.269871583549377e-03, -1.673462718931978e-03, -6.373727735573782e-02, -1.709645276825436e-01, -6.373278043997838e-02, -3.658347833389616e+00, -6.077657132320150e+00, -9.529576558049081e+02, -1.974772165696188e+01, -1.489630740746136e+02, -6.505137470526441e+07, 1.640509980084078e+03, -1.053724235364856e+04, 1.636081075560331e+03, 5.986352925218272e+08, -2.518602700655724e+09, 4.058507635108696e+08])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
