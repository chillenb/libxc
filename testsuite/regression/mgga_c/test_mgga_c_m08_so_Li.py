
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_m08_so_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.926868941379279e-02, -6.765301669904124e-02, 3.915153027565611e-03, -2.816897074146782e-02, 4.127002299591316e-03, -3.948866793779360e-02, -9.799458568968517e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_m08_so_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.567224565485152e-01, -2.565184643358887e-01, -6.072934587860201e-02, -6.058907145926259e-02, 1.279032379301734e-01, 1.278937425553115e-01, -3.795201547984248e-02, -1.372854749612784e-01, 1.498355811856024e-02, 1.598522322375177e+00, -4.962524793389993e-02, -5.018222739426198e-02, -1.152805453887996e-03, -1.691580494713458e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m08_so_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.122939676000792e-04, 4.245879352001583e-04, 2.122939676000792e-04, 1.868395648360252e-04, 3.736791296720502e-04, 1.868395648360252e-04, 1.367122744330096e-02, 2.734245488660193e-02, 1.367122744330096e-02, 1.579468355612160e+01, 3.158936711224320e+01, 1.579468355612160e+01, 4.945551781390289e+01, 9.891103562780576e+01, 4.945551781390289e+01, -7.036303926332064e-04, -1.407260785298170e-03, -7.036303926332064e-04, -6.742208881562953e-06, -1.348443912624039e-05, -6.742208881562953e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m08_so_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_m08_so_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_m08_so", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-5.690472036351175e-03, -5.690472036351058e-03, -1.184813746794633e-02, -1.184813746794633e-02, -3.269306865817574e-02, -3.269306865817447e-02, -6.953632445497013e-01, -6.953632445495548e-01, -2.663631823791794e-01, -2.663631821955414e-01, -1.356977197744148e-07, -1.356977196584878e-07, -3.569246435311061e-19, -3.585789871859474e-19]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
